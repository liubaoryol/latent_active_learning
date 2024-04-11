from datetime import datetime
import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
import wandb
import os

from latent_active_learning.scripts.config.train_hbc import train_hbc_ex
from latent_active_learning.scripts.utils import get_demos
from latent_active_learning.oracle import Oracle
from latent_active_learning.collect import get_dir_name


@train_hbc_ex.automain
def main(
    env_name,
    filter_state_until,
    kwargs,
    num_demos_train=100,
    num_demos_test=100,
    n_targets=None,
    n_epochs=None,
    use_wandb=None,
):

    rollouts, options = get_demos(num_demos=num_demos_train)
    rollouts_test, options_test = get_demos(num_demos=num_demos_test)
    gini = Oracle(
        expert_trajectories=rollouts,
        true_options=options,
        expert_trajectories_test=rollouts_test,
        true_options_test=options_test
    )
    path = get_dir_name(env_name, kwargs).split('/')[1]
    gini.save('./expert_trajs/{}'.format(path))
    # print stats
    gini.stats()