import numpy as np
import warnings
import gymnasium as gym
import os

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from latent_active_learning.wrappers.latent_wrapper import FilterLatent
from imitation.data import rollout
import imitation


def get_dir_name(env_name, kwargs):
    experts_path = 'latent_active_learning/expert_params/'
    size = kwargs['size']
    n_targs = kwargs['n_targets']
    fixed = 'fixed' if kwargs['fixed_targets'] is not None else ''
    return f'{experts_path}{env_name}_size_{size}_n-targets_{n_targs}{fixed}'

def get_environment(
    env_name,
    full_obs,
    n_envs=4,
    kwargs:dict=None):

    if full_obs:
        wrapper=None
        wrapper_kwargs=None
    else:
        wrapper = FilterLatent
        wrapper_kwargs={'unobservable_states': [-1]}

    if env_name in ['BoxWorld-v0', 'BoxWorld-continuous-v0']:
        vec_env = make_vec_env(env_name,
                               n_envs=n_envs,
                               env_kwargs=kwargs,
                               wrapper_class=wrapper,
                               wrapper_kwargs=wrapper_kwargs)

    else:
        vec_env = make_vec_env(env_name,
                               n_envs=n_envs,
                               wrapper_class=wrapper,
                               wrapper_kwargs=wrapper_kwargs
                               )
    return vec_env


def filter_TrajsWRewards(rollouts, filter_until=-1):
    rollouts_filtered = []
    for rollout in rollouts:
        filtered = imitation.data.types.TrajectoryWithRew(
            obs = rollout.obs[:,:filter_until],
            acts = rollout.acts,
            infos = rollout.infos,
            terminal = rollout.terminal,
            rews = rollout.rews
        )
        rollouts_filtered.append(filtered)
    return rollouts_filtered


def get_trajectories(
        env_name,
        model,
        full_obs=True,
        n_demo=10,
        seed=42,
        kwargs: dict=None
        ):
    vec_env = get_environment(env_name=env_name,
                              full_obs=True,
                              kwargs=kwargs)

    rollouts = rollout.rollout(
        model,
        vec_env,
        rollout.make_sample_until(min_episodes=n_demo),
        rng=np.random.default_rng(seed),
        unwrap=False
    )

    if not full_obs:
        rollouts = filter_TrajsWRewards(rollouts)

    return rollouts

def get_expert(env_name, kwargs, n_epoch=1e6):
    expert_dir = get_dir_name(env_name, kwargs)
    if os.path.exists(f'{expert_dir}.zip'):
        # Load trained policy
        expert_model_dir = get_dir_name(env_name, kwargs)
        expert = PPO.load(expert_model_dir)
        return expert
    
    warnings.warn("Expert has not been trained on this environment. Training an expert...")
    vec_env = get_environment(env_name, full_obs=True, kwargs=kwargs)
    model = PPO("MlpPolicy", vec_env, verbose=1)
    model.learn(total_timesteps=n_epoch, progress_bar=True)
    model.save(expert_dir)
    return model