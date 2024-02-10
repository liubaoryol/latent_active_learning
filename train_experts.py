import torch
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

import latent_active_learning
from latent_active_learning.baselines.OptionGAIL.utils.config import ARGConfig
from latent_active_learning.baselines.OptionGAIL.default_config import mujoco_config

if __name__ == "__main__":

    arg = ARGConfig()
    arg.add_arg("env_name", "BoxWorld-v0", "Environment name: BoxWorld-v0, Walker2d-v4, Hopper-v4")
    arg.add_arg("device", "cuda:0", "Computing device")
    arg.add_arg("tag", "default", "Experiment tag")
    arg.add_arg("n_epoch", 1e7, "Number of training epochs")
    arg.add_arg("seed", torch.randint(100, ()).item(), "Random seed")
    arg.parser()
    #NOTE: when training PPO, we assume full observability.

    config = mujoco_config
    config.update(arg)

    env_name = arg.env_name
    # Create parallel environments
    # vec_env = make_vec_env(env_name, n_envs=4, env_kwargs={'n_targets': 2})


    if env_name == 'BoxWorld-v0':
        vec_env = make_vec_env(env_name, n_envs=4, env_kwargs={'n_targets': 2})
    else:
        vec_env = make_vec_env(env_name, n_envs=4)

        
    model = PPO("MlpPolicy", vec_env, verbose=1)
    model.learn(total_timesteps=config.n_epoch, progress_bar=True)
    model.save('latent_active_learning/expert_params/' + env_name + '_expert')

    # del model # remove to demonstrate saving and loading

    model = PPO.load('latent_active_learning/expert_params/' + env_name + '_expert')

    if env_name == 'BoxWorld-v0':
        box = gym.make(env_name, n_targets=2, render_mode='human')
    else:
        box = gym.make(env_name, render_mode='human')

    box.reset()
    while True:
        print('ji')
        action = np.random.randint(4)
        box.step(action)

    obs = box.reset()[0]
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, terminated, info = box.step(action.tolist())
