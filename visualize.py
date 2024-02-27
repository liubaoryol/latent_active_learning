
# VISUALIZE:
import time
import gymnasium as gym

import latent_active_learning
from latent_active_learning.wrappers.latent_wrapper import TransformBoxWorldReward, FilterLatent


kwargs = {
    'size': 5,
    'n_targets': 2,
    'allow_variable_horizon': True,
    'fixed_targets': [[0,0],[4,4]], # NOTE: Remember to change size to 5
    # 'fixed_targets': [[0,0],[4,4], [0,4]] # NOTE: Remember to change size to 5
    # 'fixed_targets': [[0,0], [9,0] ,[0,9],[9,9]] # NOTE: Remember to change size to 10
    # 'fixed_targets': [[0,0],[4,4], [0,4], [9,0], [0,9],[9,9]] # NOTE: Remember to change size to 10
    'render_mode': 'human'
    }

env = gym.make("BoxWorld-v0", **kwargs)
wrapped_env = TransformBoxWorldReward(FilterLatent(env, [-1]))
observation, info = wrapped_env.reset()
state = None
rewards = []
returns = []
for _ in range(100):
    action, state = hbc.predict([observation], state, episode_start=False)  # agent policy that uses the observation and info
    print(state)
    # time.sleep(1)
    observation, reward, terminated, truncated, info = wrapped_env.step(action[0])
    rewards.append(reward)
    if terminated or truncated:
        returns.append(np.sum(rewards))
        rewards = []
        observation, info = wrapped_env.reset()
        state = None

wrapped_env.close()