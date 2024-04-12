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
    experts_path = 'expert_params/'
    size = kwargs['size']
    n_targs = kwargs['n_targets']
    return f'{experts_path}{env_name[:-3]}-size{size}-targets{n_targs}'

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
    for roll in rollouts:
        filtered = imitation.data.types.TrajectoryWithRew(
            obs = roll.obs[:,:filter_until],
            acts = roll.acts,
            infos = roll.infos,
            terminal = roll.terminal,
            rews = roll.rews
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
    model = train_expert(env_name, kwargs, n_epoch)
    model.save(expert_dir)
    return model

def train_expert(env_name, kwargs, n_epoch=1e6):
    n_epoch = 1e7
    vec_env = get_environment(env_name, full_obs=True, kwargs=kwargs)
    batch_size = 1024
    rollout_buffer_size = 10240
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        batch_size = batch_size,
        n_steps = rollout_buffer_size,
        ent_coef = 0.2,
        learning_rate = 0.01
        )
    model.learn(total_timesteps=n_epoch, progress_bar=True)
    return model

#     env_name = "BoxWorld-v0"
#     n_epochs = 100
#     use_wandb=True
#     num_demos=100
#     targets = [
#         [ 1, 1 ],
#         [ 7, 1 ],
#         [ 4, 6 ],
#         [ 5, 0 ],
#         [ 7, 3 ],
#         [ 6, 8 ],
#     ]
#     danger = [
#         [ 1, 1 ],
#     ]
#     obstacles = [
#         [ 2, 0 ],
#         [ 8, 2 ],
#         [ 7, 4 ],
#         [ 3, 7 ],
#         [ 4, 7 ],
#         [ 6, 7 ],
#         [ 0, 4 ],
#         [ 0, 5 ],
#         [ 1, 2 ],
#         [ 1, 5 ],
#         [ 1, 8 ],
#         [ 2, 3 ],
#         [ 2, 8 ],
#         [ 3, 1 ],
#         [ 3, 5 ],
#         [ 3, 6 ],
#         [ 4, 1 ],
#         [ 4, 2 ],
#         [ 4, 5 ],
#         [ 4, 8 ],
#         [ 5, 5 ],
#         [ 5, 8 ],
#         [ 6, 1 ],
#         [ 6, 2 ],
#         [ 6, 3 ],
#         [ 7, 0 ],
#         [ 7, 2 ],
#         [ 7, 6 ],
#         [ 7, 8 ],
#         [ 8, 8 ],
#         [ 9, 4 ],
#         [ 9, 5 ],
#     ]

# vec_env.envs[3].unwrapped.danger_reward = -200