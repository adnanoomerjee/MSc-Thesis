from brax import base
import jax.numpy as jp

def reward(env, obstacles_complete, state: base.State):

    next_obstacle_coord = env.obstacle_completion_coords[obstacles_complete.astype(int)]
    passed_next_obstacle = state.x.pos[0, 1] >= next_obstacle_coord

    def sparse_reward():
      return env.reward_milestone * passed_next_obstacle - (1.0 - passed_next_obstacle)

    def dense_reward():
      if env.reward_movement == 'position':
        return 62.0 - state.x.pos[0, 1]
      else:
        return state.xd.vel[0, 1]
    
    if env.reward_type == 'dense':
      reward = dense_reward()
    else:
      reward = sparse_reward()

    obstacles_complete = passed_next_obstacle * (obstacles_complete + 1) + (1 - passed_next_obstacle) * obstacles_complete
    return reward, obstacles_complete
    