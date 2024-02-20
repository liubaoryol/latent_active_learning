
# VISUALIZE:
import gymnasium as gym

import latent_active_learning
from latent_active_learning.wrappers.latent_wrapper import TransformBoxWorldReward, FilterLatent


kwargs = {
    'size': 5,
    'n_targets': 3,
    'allow_variable_horizon': True,
    'fixed_targets': [[0,0],[4,4], [0,4]]
    }

env = gym.make("BoxWorld-v0", **kwargs)
wrapped_env = TransformBoxWorldReward(FilterLatent(env, [-1]))
observation, info = wrapped_env.reset()
state = None

for _ in range(100):
    action, state = hbc.predict([observation], state, episode_start=False)  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = wrapped_env.step(action[0])
    if terminated or truncated:
        observation, info = env.reset()
        state = None

env.close()