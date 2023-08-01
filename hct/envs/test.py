import jax
from hct.envs.tools import timeit

from brax.envs import Env

def test(env: Env, iterations: int):

    reset_states = []
    step_states = []
    reset_times = []

    rng = jax.random.PRNGKey(0)
    rng, rng1, rng2 = jax.random.split(rng, 3)

    action = jax.random.uniform(rng1, shape=(env.sys.act_size(), 1), minval=-1, maxval=1)

    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    reset_state, reset_time = timeit(jit_reset, rng2)
    step_state, step_time = timeit(jit_step, reset_state, action)

    time_to_jit_reset = reset_time
    time_to_jit_step = step_time

    time_to_call_reset = 0
    time_to_call_step = 0

    for i in range(iterations):

      rng, rng1, rng2= jax.random.split(rng, 3)

      action = jax.random.uniform(rng1, shape=(env.sys.act_size(), 1), minval=-1, maxval=1)

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