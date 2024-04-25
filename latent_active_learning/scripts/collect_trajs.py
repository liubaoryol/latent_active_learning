from datetime import datetime
import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
import wandb
import os

from latent_active_learning.scripts.config.train_hbc import train_hbc_ex
from latent_active_learning.scripts.utils import get_demos, get_movers_demos
from latent_active_learning.oracle import Oracle
from latent_active_learning.collect import get_dir_name


@train_hbc_ex.automain
def main(
    env_name,
    filter_state_until=None,
    kwargs=None,
    num_demos_train=50,
    num_demos_test=50,
    n_targets=None,
    n_epochs=None,
    use_wandb=None,
    movers_optimal=False,
    options_w_robot=False,
    state_w_robot_opts=False,
    fixed_latent=False
):
    if env_name=='EnvMovers-v0':
        
        rollouts, options = get_movers_demos(num_demos_train,
                                             movers_optimal,
                                             options_w_robot,
                                             state_w_robot_opts,
                                             fixed_latent
                                             )
        rollouts_test, options_test = get_movers_demos(num_demos_test,
                                                       movers_optimal,
                                                       options_w_robot,
                                                       state_w_robot_opts,
                                                       fixed_latent
                                                       )
        path = 'EnvMovers{}{}{}{}'.format(
            '-optimal' if movers_optimal else '',
            '-options-include-robot' if options_w_robot else '',
            '-state-include-robot' if state_w_robot_opts else '',
            '-fixed-latent' if fixed_latent else ''
            )
    else:
        rollouts, options = get_demos(num_demos=num_demos_train)
        rollouts_test, options_test = get_demos(num_demos=num_demos_test)
        path = get_dir_name(env_name, kwargs).split('/')[1]

    gini = Oracle(
        expert_trajectories=rollouts,
        true_options=options,
        expert_trajectories_test=rollouts_test,
        true_options_test=options_test
    )
    
    gini.save('./expert_trajs/{}'.format(path))
    # print stats
    gini.stats()