from hct.training.training_run import *
import time

import functools
import time
from typing import Callable, Optional, Tuple, Union

from absl import logging
from brax import envs
from brax.training import gradients
from brax.training import pmap
from hct.training import types
from brax.training.acme import running_statistics
from brax.training.acme import specs
from brax.training.types import Params
from brax.training.types import PRNGKey
from brax.training.types import NetworkFactory
from brax.v1 import envs as envs_v1

from hct.training import network_factory as ppo_networks
from hct.envs.wrappers.training import wrap as wrap_for_training
from hct.training import acting
from hct.training import losses as ppo_losses
from hct import envs as hct_envs
from hct.io.model import save

import flax
import jax
import jax.numpy as jnp
import optax


InferenceParams = Tuple[running_statistics.NestedMeanStd, Params]
Metrics = types.Metrics

_PMAP_AXIS_NAME = 'i'

  
def detect_nan_in_pytree(pytree):
    nan_tree = jax.tree_map(lambda x: jnp.any(jnp.isnan(x)), pytree)
    return jax.tree_util.tree_reduce(jnp.logical_or, nan_tree)


@flax.struct.dataclass
class TrainingState:
  """Contains training state for the learner."""
  optimizer_state: optax.OptState
  params: ppo_losses.PPONetworkParams
  normalizer_params: running_statistics.RunningStatisticsState
  env_steps: jnp.ndarray


def _unpmap(v):
  return jax.tree_util.tree_map(lambda x: x[0], v)


def _strip_weak_type(tree):
  # brax user code is sometimes ambiguous about weak_type.  in order to
  # avoid extra jit recompilations we strip all weak types from user input
  def f(leaf):
    leaf = jnp.asarray(leaf)
    return leaf.astype(leaf.dtype)
  return jax.tree_util.tree_map(f, tree)


def train(environment: Union[envs_v1.Env, envs.Env], # Training enviroment
          num_timesteps: int, # Number of timesteps to train
          num_envs: int = 1, 
          max_devices_per_host: Optional[int] = None,
          num_eval_envs: int = 128,
          learning_rate: float = 1e-4, # α
          entropy_cost: float = 1e-4,
          discounting: float = 0.9, # γ
          seed: int = 0,
          episode_length = None,
          unroll_length: int = 10, # Number of timesteps the agent collects before it updates its policy
          batch_size: int = 32, # Number of experiences (state, action, reward, next state) sampled from the environment per training batch.
          num_minibatches: int = 16, # Number of different subsets that the batch of experiences is divided into. PPO typically uses minibatches to compute an estimate of the expected return and update the policy.
          num_updates_per_batch: int = 2,
          gradient_clipping: float = 0.1,
          num_evals: int = 1, # Number of evaluations intended to be performed, one per epoch. Evaluations refer to the times when the agent's performance is assessed, usually without learning from this interaction with the environment. 
          normalize_observations: bool = False,
          reward_scaling: float = 1.,
          clipping_epsilon: float = .3,
          gae_lambda: float = .95,
          deterministic_eval: bool = False,
          network_factory: NetworkFactory[
              ppo_networks.PPONetworks] = ppo_networks.make_ppo_networks, # Sets network_factory as a function that makes PPONetwork objects
          progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
          normalize_advantage: bool = True,
          eval_env: Optional[envs.Env] = None,
          policy_params_fn: Callable[..., None] = lambda *args: None):
  
  """PPO training."""
  assert batch_size * num_minibatches % num_envs == 0
  xt = time.time()

  # Devices and processes setup
  process_count = jax.process_count() # no. of jax processes in backend
  process_id = jax.process_index() # Returns the integer process index of this process.
  local_device_count = jax.local_device_count() # Returns the number of devices addressable by this process.
  local_devices_to_use = local_device_count
  if max_devices_per_host:
    local_devices_to_use = min(local_devices_to_use, max_devices_per_host)
  if 'gpu' in str(jax.devices()[0]).lower():
    logging.info("JAX is running on GPU.")
  else:
    logging.info("JAX is not running on GPU.")
  logging.info(
      'Device count: %d, process count: %d (id %d), local device count: %d, '
      'devices to be used count: %d', jax.device_count(), process_count,
      process_id, local_device_count, local_devices_to_use)
  device_count = local_devices_to_use * process_count

  env = environment
  episode_length = env.episode_length if episode_length is None else episode_length
  action_repeat = env.action_repeat

  if hasattr(env, 'horizon'):
    discounting = 1 - 1/env.horizon

  # Number of environment steps executed for every training step.
  env_step_per_training_step = (
      batch_size * unroll_length * num_minibatches * action_repeat)
  num_evals_after_init = max(num_evals - 1, 1)

  # Number of training_step calls per training_epoch call.
      # equals to ceil(num_timesteps / (num_evals * env_step_per_training_step))
      # Num_epochs = num_evals_after_init
      # num_training_steps = num_timesteps / env_step_per_training_step
  num_training_steps_per_epoch = -(
      -num_timesteps // (num_evals_after_init * env_step_per_training_step))


  # Initialise keys
  key = jax.random.PRNGKey(seed)
  global_key, local_key = jax.random.split(key)
  del key
  local_key = jax.random.fold_in(local_key, process_id) # mixes an integer (here, process_id) into a PRNG key to produce a new key.
  local_key, key_env, eval_key = jax.random.split(local_key, 3)
  # key_networks should be global, so that networks are initialized the same way for different processes.
  key_policy, key_value = jax.random.split(global_key)
  del global_key

  
  # Wrap environment using function from envs.wrappers.training:
      # 1. Wraps an environment using wrap.EpisodeWrapper, which defines steps and resets in the environment according to action repeats.
      # 2. Wraps with wrap.VmapWrapper, vectorising resets and steps using vmap.
      # 3. Wraps with wrap.AutoResetWrapper, automatically resetting environment when episode is done.
  assert num_envs % device_count == 0

  env = wrap_for_training(
      env, episode_length=episode_length, action_repeat=action_repeat)


  # Resets environments
  reset_fn = jax.jit(jax.vmap(env.reset)) 
  key_envs = jax.random.split(key_env, num_envs // process_count) # Sets key_envs
  key_envs = jnp.reshape(key_envs,
                         (local_devices_to_use, -1) + key_envs.shape[1:])
  env_state = reset_fn(key_envs)


  # Define function for normalizing observations
  normalize = lambda x, y: x
  if normalize_observations:
    normalize = running_statistics.normalize

  
  #Define network
  ppo_network = network_factory(
      env=env,
      observation_size=env_state.obs.shape[-1],
      preprocess_observations_fn=normalize
      )
  
  make_inference_fn = ppo_networks.make_inference_fn
  make_policy = ppo_networks.make_inference_fn(ppo_network)

  #Define optimizer
  optimizer = optax.adam(learning_rate=learning_rate)
  if int(gradient_clipping) != 0:
    optimizer = optax.chain(
      optax.clip(gradient_clipping), optimizer)


  #Define loss function
  loss_fn = functools.partial(
      ppo_losses.compute_ppo_loss,
      env=env,
      ppo_network=ppo_network,
      entropy_cost=entropy_cost,
      discounting=discounting,
      reward_scaling=reward_scaling,
      gae_lambda=gae_lambda,
      clipping_epsilon=clipping_epsilon,
      normalize_advantage=normalize_advantage)


  # Define gradient update (change in policy parameters)
  gradient_update_fn = gradients.gradient_update_fn(
      loss_fn, optimizer, pmap_axis_name=_PMAP_AXIS_NAME, has_aux=True)


  def minibatch_step(
      carry, data: types.Transition,
      normalizer_params: running_statistics.RunningStatisticsState):
    optimizer_state, params, key = carry
    key, key_loss = jax.random.split(key)
    (_, metrics), params, optimizer_state = gradient_update_fn(
        params,
        normalizer_params,
        data,
        key_loss,
        optimizer_state=optimizer_state)

    return (optimizer_state, params, key), metrics


  # Gradient descent update, includes minibatch_step
  def sgd_step(carry, unused_t, data: types.Transition,
               normalizer_params: running_statistics.RunningStatisticsState):
    optimizer_state, params, key = carry
    key, key_perm, key_grad = jax.random.split(key, 3)

    def convert_data(x: jnp.ndarray):
      x = jax.random.permutation(key_perm, x)
      x = jnp.reshape(x, (num_minibatches, -1) + x.shape[1:])
      return x

    shuffled_data = jax.tree_util.tree_map(convert_data, data)
    (optimizer_state, params, _), metrics = jax.lax.scan(
        functools.partial(minibatch_step, normalizer_params=normalizer_params),
        (optimizer_state, params, key_grad),
        shuffled_data,
        length=num_minibatches)
    return (optimizer_state, params, key), metrics

  # Full training step, includes sgd_step
  def training_step(
      carry: Tuple[TrainingState, envs.State, PRNGKey],
      unused_t) -> Tuple[Tuple[TrainingState, envs.State, PRNGKey], Metrics]:
    training_state, state, key = carry
    key_sgd, key_generate_unroll, new_key = jax.random.split(key, 3)

    policy = make_policy(
        (training_state.normalizer_params, training_state.params.policy),
        obs_mask=env.obs_mask,
        action_mask=env.action_mask,
        non_actuator_nodes=env.non_actuator_nodes) # (processor_params, policy_params)

    def f(carry, unused_t):
      current_state, current_key = carry
      current_key, next_key = jax.random.split(current_key)
      next_state, data = acting.generate_unroll(
          env,
          current_state,
          policy,
          current_key,
          unroll_length,
          extra_fields=('truncation',))
      return (next_state, next_key), data
    
    (state, _), data = jax.lax.scan(
        f, (state, key_generate_unroll), (),
        length=batch_size * num_minibatches // num_envs)
    # Have leading dimensions (batch_size * num_minibatches, unroll_length)
    data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 1, 2), data)
    data = jax.tree_util.tree_map(lambda x: jnp.reshape(x, (-1,) + x.shape[2:]),
                                  data)
    assert data.discount.shape[1:] == (unroll_length,)

    # Update normalization params and normalize observations.
    normalizer_params = running_statistics.update(
        training_state.normalizer_params,
        data.observation,
        pmap_axis_name=_PMAP_AXIS_NAME)

    (optimizer_state, params, _), metrics = jax.lax.scan(
        functools.partial(
            sgd_step, data=data, normalizer_params=normalizer_params),
        (training_state.optimizer_state, training_state.params, key_sgd), (),
        length=num_updates_per_batch)

    new_training_state = TrainingState(
        optimizer_state=optimizer_state,
        params=params,
        normalizer_params=normalizer_params,
        env_steps=training_state.env_steps + env_step_per_training_step)
    return (new_training_state, state, new_key), metrics
  
  def training_epoch(training_state: TrainingState, state: envs.State,
                     key: PRNGKey) -> Tuple[TrainingState, envs.State, Metrics]:
    (training_state, state, _), loss_metrics = jax.lax.scan(
        training_step, (training_state, state, key), (),
        length=num_training_steps_per_epoch)
    loss_metrics = jax.tree_util.tree_map(jnp.mean, loss_metrics)
    return training_state, state, loss_metrics

  training_epoch = jax.pmap(training_epoch, axis_name=_PMAP_AXIS_NAME)

  # Note that this is NOT a pure jittable method.
  def training_epoch_with_timing(
      training_state: TrainingState, env_state: envs.State,
      key: PRNGKey) -> Tuple[TrainingState, envs.State, Metrics]:
    nonlocal training_walltime
    t = time.time()
    training_state, env_state = _strip_weak_type((training_state, env_state))
    result = training_epoch(training_state, env_state, key)
    training_state, env_state, metrics = _strip_weak_type(result)

    metrics = jax.tree_util.tree_map(jnp.mean, metrics)
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

    epoch_training_time = time.time() - t
    training_walltime += epoch_training_time
    sps = (num_training_steps_per_epoch *
           env_step_per_training_step) / epoch_training_time
    metrics = {
        'training/sps': sps,
        'training/walltime': training_walltime,
        **{f'training/{name}': value for name, value in metrics.items()}
    }
    return training_state, env_state, metrics

  init_params = ppo_losses.PPONetworkParams(
      policy=ppo_network.policy_network.init(key_policy),
      value=ppo_network.value_network.init(key_value))
  training_state = TrainingState(  # pytype: disable=wrong-arg-types  # jax-ndarray
      optimizer_state=optimizer.init(init_params),  # pytype: disable=wrong-arg-types  # numpy-scalars
      params=init_params,
      normalizer_params=running_statistics.init_state(
          specs.Array(env_state.obs.shape[-1:], jnp.float32)),
      env_steps=0)
  training_state = jax.device_put_replicated(
      training_state,
      jax.local_devices()[:local_devices_to_use])

  logging.info(detect_nan_in_pytree((training_state, env_state)))

  if not eval_env:
    eval_env = env
  else:
    eval_env = wrap_for_training(
        eval_env, episode_length=episode_length, action_repeat=action_repeat)

  evaluator = acting.Evaluator(
      eval_env,
      functools.partial(make_policy, 
                        train=False,
                        obs_mask=env.obs_mask, 
                        action_mask=env.action_mask,
                        non_actuator_nodes=env.non_actuator_nodes,
                        deterministic=deterministic_eval),
      num_eval_envs=num_eval_envs,
      episode_length=episode_length,
      action_repeat=action_repeat,
      key=eval_key)

  # Run initial eval
  logging.info('Running initial eval')
  if process_id == 0 and num_evals > 1:
    metrics, data = evaluator.run_evaluation(
        _unpmap(
            (training_state.normalizer_params, training_state.params.policy)),
        training_metrics={})
    logging.info(metrics)
    logging.info(detect_nan_in_pytree((data)))
    save("test/data", (data))

    progress_fn(0, metrics)

  training_walltime = 0
  current_step = 0
  for it in range(num_evals_after_init):
    logging.info('starting iteration %s/%s, %s steps, %s', it + 1, num_evals_after_init, current_step, time.time() - xt)

    # optimization
    epoch_key, local_key = jax.random.split(local_key)
    epoch_keys = jax.random.split(epoch_key, local_devices_to_use)
    (training_state, env_state,
     training_metrics) = training_epoch_with_timing(training_state, env_state,
                                                    epoch_keys)
    if detect_nan_in_pytree((training_state, env_state,
     training_metrics)):
      break
    current_step = int(_unpmap(training_state.env_steps))
    if process_id == 0:
      # Run evals.
      metrics, data = evaluator.run_evaluation(
          _unpmap(
              (training_state.normalizer_params, training_state.params.policy)),
          training_metrics)
      logging.info(metrics)
      progress_fn(current_step, metrics)
      params = _unpmap(
          (training_state.normalizer_params, training_state.params.policy))
      
      if not (it % (num_evals_after_init//10)) or it == num_evals_after_init-1:
        savetime = time.time()
        policy_params_fn(current_step, make_policy, params, make_inference_fn, ppo_network)
        logging.info('saving model, time to save: %s', time.time() - savetime)

  total_steps = current_step
  assert total_steps >= num_timesteps

  # If there was no mistakes the training_state should still be identical on all
  # devices.
  pmap.assert_is_replicated(training_state)
  params = _unpmap(
      (training_state.normalizer_params, training_state.params.policy))
  logging.info('total steps: %s', total_steps)
  pmap.synchronize_hosts()
  return (make_policy, params, metrics)

train_fn = functools.partial(train,  
    num_timesteps=50_000_000, 
    num_envs=2048, 
    max_devices_per_host=None,
    learning_rate=3e-4, 
    entropy_cost=1e-2,
    discounting=0.9, 
    gradient_clipping=0.1,
    seed=1,
    unroll_length=5,
    batch_size=256, 
    num_minibatches=32,
    num_updates_per_batch=4,
    num_evals=512, 
    normalize_observations=False,
    reward_scaling=10,
    clipping_epsilon=.3,
    gae_lambda=.95,
    deterministic_eval=False,
    normalize_advantage=True
)

def get_train_fn(env, **kwargs):
  """Returns a train function.

  Args:
    env: environment
    **kwargs: keyword arguments that get passed to the Env class constructor

  Returns:
    train_fn
  """
  return functools.partial(train_fn, environment=env, **kwargs)


def training_run(env_name, env_parameters, train_parameters, variant_name, filepath):
    
    current_datetime = datetime.now().strftime("%Y-%m-%d, %H:%M:%S")
    env_parameters_name = env_parameters.copy()
    env_parameters_name.pop('architecture_configs')

    training_run_name = f'{variant_name}'
    filepath = f'{filepath}{training_run_name}/'
    
    #if os.path.exists(filepath):
    #  shutil.rmtree(filepath)
    os.makedirs(os.path.dirname(filepath))      

    logging.get_absl_handler().use_absl_log_file('log', filepath)

    env = hct_envs.get_environment(
        env_name=env_name,
        **env_parameters)
    
    train_fn = get_train_fn(
        env=env,
        **train_parameters)

    training_run_metrics = {
      'model_variant_name': training_run_name
    }
    training_parameters = train_fn.keywords.copy()
    training_parameters.pop('environment')
    training_parameters['env_name'] = env_name

    model.save(obj=training_parameters, path=f'{filepath}/training_params')
    model.save(obj=env_parameters, path=f'{filepath}/env_params')
    model.save(obj=env, path=f'{filepath}/env')

    def progress(num_steps, metrics):
      training_run_metrics.update({str(num_steps): metrics})

    def save(current_step, make_policy, params, make_inference_fn, network):
      model.save(obj=training_run_metrics, path=f'{filepath}/training_metrics')
      model.save(obj=params, path=f'{filepath}/model_params')
      model.save(obj=make_inference_fn, path=f'{filepath}/make_inference_fn')
      model.save(obj=network, path=f'{filepath}/network')
      
    make_policy, params, metrics = train_fn(
        progress_fn=progress, 
        policy_params_fn=save
    )

class PrintLogger:
    def __init__(self, stdout, log_level=logging.INFO):
        self.stdout = stdout
        self.log_level = log_level

    def write(self, message):
        # Avoid logging empty messages
        if message.strip() != "":
            logging.log(self.log_level, message.strip())
        self.stdout.write(message)

    def flush(self):
        self.stdout.flush()

# Set up the absl logging and verbosity level
logging.set_verbosity(logging.INFO)

# Redirect stdout to our custom logger
sys.stdout = PrintLogger(sys.stdout)

low_level_modelpath = "/nfs/nhome/live/aoomerjee/MSc-Thesis/hct/training/hyperparam_sweeps/hma/low_level/runs/2"

PARAMS = {
    'low_level_modelpath': low_level_modelpath,
    'reward_type': 'dense',
    'reward_movement':  'velocity',
    'action_repeat': 1,
    'architecture_configs': VLARGE_MLP_CONFIGS
}

TRAINING_PARAMETERS = {
    'num_timesteps':50_000_000, 
    'num_envs':2048, 
    'max_devices_per_host':None,
    'learning_rate':3e-4, 
    'entropy_cost':1e-2,
    'discounting':0.99, # trial 0.97 vs 0.99, 0.97 better
    'gradient_clipping':1.0, # trial 1.0 vs 1000000, 1.0 better
    'seed':3,
    'unroll_length':5,
    'batch_size':2048, 
    'num_minibatches':32,
    'num_updates_per_batch':4,
    'num_evals':102, 
    'normalize_observations':True,
    'reward_scaling':10,
    'clipping_epsilon':.3,
    'gae_lambda':.95,
    'deterministic_eval':False,
    'normalize_advantage':True}

#jax.config.update("jax_debug_nans", True)
logging.set_verbosity(logging.INFO)
env_name = 'HMA2GapsEnv'


def main(argv):
  training_run(env_name, PARAMS, TRAINING_PARAMETERS, f'test {time.time()}', 'test/')

if __name__== '__main__':
  app.run(main)
