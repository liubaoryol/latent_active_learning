import numpy as np
import torch
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from latent_active_learning.wrappers.latent_wrapper import FilterLatent
from imitation.data import rollout
import imitation

def get_environment(env_name, full_obs, n_envs=4):

    if full_obs:
        wrapper=None
        wrapper_kwargs=None
    else:
        wrapper = FilterLatent
        wrapper_kwargs={'unobservable_states': [-1]}

    if env_name in ['BoxWorld-v0', 'BoxWorld-continuous-v0']:
        vec_env = make_vec_env(env_name,
                               n_envs=n_envs,
                               env_kwargs={
                                'n_targets': 2,
                                'allow_variable_horizon': True
                                },
                               wrapper_class=wrapper,
                               wrapper_kwargs=wrapper_kwargs)

    else:
        vec_env = make_vec_env(env_name,
                               n_envs=n_envs,
                               wrapper_class=wrapper,
                               wrapper_kwargs=wrapper_kwargs
                               )
    return vec_env


def filter_intent_TrajsWRewards(rollouts):
    rollouts_filtered = []
    for rollout in rollouts:
        filtered = imitation.data.types.TrajectoryWithRew(
            obs = rollout.obs[:,:-1],
            acts = rollout.acts,
            infos = rollout.infos,
            terminal = rollout.terminal,
            rews = rollout.rews
        )
        rollouts_filtered.append(filtered)
    return rollouts_filtered


def get_expert_trajectories(env_name, n_demo=10, full_obs=True, seed=42):
    vec_env = get_environment(env_name, True)

    # Load trained policy and get expert trajectories
    expert_model_dir = 'latent_active_learning/expert_params/' + env_name + '_expert'
    expert = PPO.load(expert_model_dir)
    rollouts = rollout.rollout(
        expert,
        vec_env,
        rollout.make_sample_until(min_episodes=n_demo),
        rng=np.random.default_rng(seed),
        unwrap=False
    )

    if not full_obs:
        rollouts = filter_intent_TrajsWRewards(rollouts)

    return rollouts


def get_expert(env_name, n_demo=10, full_obs=True):
    # Load trained policy and get expert trajectories
    expert_model_dir = 'latent_active_learning/expert_params/' + env_name + '_expert'
    expert = PPO.load(expert_model_dir)

    return expert

def train_expert(env_name, n_epoch=1e6):
    vec_env = get_environment(env_name, True)

    model = PPO("MlpPolicy", vec_env, verbose=1)
    model.learn(total_timesteps=n_epoch, progress_bar=True)
    model.save('latent_active_learning/expert_params/' + env_name + '_expert')
