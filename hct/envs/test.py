import jax
from hct.envs.tools import timeit
from hct.io.html import render
from hct.io.model import load

from brax.envs import Env

def test(env: Env, iterations: int = 1, jit = True):
    
    action_shape = env.action_shape

    reset_states = []
    step_states = []
    reset_times = []

    rng = jax.random.PRNGKey(0)
    rng, rng1, rng2 = jax.random.split(rng, 3)

    action = jax.random.uniform(rng1, shape=action_shape, minval=-1, maxval=1)

    if jit:
      jit_reset = jax.jit(env.reset)
      jit_step = jax.jit(env.step)
    else:
      jit_reset = env.reset
      jit_step = env.step

    reset_state, reset_time = timeit(jit_reset, rng2)
    step_state, step_time = timeit(jit_step, reset_state, action)

    time_to_jit_reset = reset_time
    time_to_jit_step = step_time

    time_to_call_reset = 0
    time_to_call_step = 0

    for i in range(iterations):

      rng, rng1, rng2= jax.random.split(rng, 3)

      action = jax.random.uniform(rng1, shape=action_shape, minval=-1, maxval=1)

      reset_state, reset_time = timeit(jit_reset, rng2)
      step_state, step_time = timeit(jit_step, reset_state, action)
      
      time_to_call_reset += reset_time/iterations
      time_to_call_step += step_time/iterations

      reset_states.append(reset_state)
      step_states.append(step_state)

      reset_times.append(reset_time)
      
    print(f"Time to JIT 'reset': {time_to_jit_reset} seconds")
    print(f"Time to JIT 'step': {time_to_jit_step} seconds")
    print(f"Time to call 'reset' after JIT compilation: {time_to_call_reset} seconds")
    print(f"Time to call 'step' after JIT compilation: {time_to_call_step} seconds")

    return jit_reset, jit_step, reset_states, step_states, reset_times

def rollout(modelpath, seed = 1):
  
    env = load(f"{modelpath}/env")

    network = load(f"{modelpath}/network")
    params = load(f"{modelpath}/model_params")
    make_inference_fn = load(f"{modelpath}/make_inference_fn")

    policy = make_inference_fn(network)(
        params=params,
        obs_mask=env.obs_mask, 
        action_mask=env.action_mask,
        non_actuator_nodes=env.non_actuator_nodes,
        deterministic=False,
        train=False
        )

    states = []

    rng = jax.random.PRNGKey(seed)

    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    jit_policy = jax.jit(policy)

    state = jit_reset(rng)

    for _ in range(env.episode_length):
      states.append(state)
      rng, rng1 = jax.random.split(rng)
      action, _ = jit_policy(state.obs, rng1)
      state = jit_step(state, action)

    return render(env.sys.replace(dt=env.dt), [state.pipeline_state for state in states]), states, env