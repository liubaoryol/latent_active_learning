import os
from datetime import datetime
import numpy as np
import gymnasium as gym
from stable_baselines3.common.monitor import Monitor

import imitation
from imitation.algorithms import bc
from imitation.data import rollout
from imitation.util import logger as imit_logger
from latent_active_learning.scripts.utils import get_demos
from latent_active_learning.scripts.config.train_hbc import train_hbc_ex


timestamp = lambda: datetime.now().strftime('%m-%d-%Y_%H-%M-%S')

def set_logger(exp_identifier):
    CURR_DIR = os.getcwd()
    timestamp = lambda: datetime.now().strftime('%m-%d-%Y_%H-%M-%S')

    logging_dir = os.path.join(
        CURR_DIR,
        f'results/{exp_identifier}_{timestamp()}/'
        )
    logger = imit_logger.configure(
        logging_dir,
        ["stdout", "csv", "tensorboard", "wandb"]
        )
    return logger


# Supervised case == query_percent = 1
# UNSUPERVISED BC == query_percent = 0
# CHECK HOW BC WORKS WITH THE OPTIONS ESTIMATED BY AN UNINITIALIZED HBC


@train_hbc_ex.automain
def main(_config,
         env_name,
         n_targets,
         filter_state_until,
         kwargs,
         n_epochs,
         use_wandb,
         query_percent):
    
    assert query_percent in [0, 1], '`query_percent` must equal either 0 or 1'

    # if use_wandb:
    #     import wandb
    #     run = wandb.init(
    #         project=f'{env_name[:-3]}-size{kwargs["size"]}-targets{n_targets}',
    #         name='BehavioralCloning_{}{}_{}'.format(
    #             'queryCap' if query_cap is not None else 'queryPercent',
    #             query_cap if query_cap is not None else query_percent,
    #             timestamp()
    #         ),
    #         tags=['bc'],
    #         config=_config,
    #         monitor_gym=True, # NOTE: had to make changes to an __init__ file to make this work. I'm not sure if it will work
    #         save_code=True
    #     )
    # else:
    #     run = None
    
    rollouts, options = get_demos()

    if query_percent:
        for idx, demo in enumerate(rollouts):
            rollouts[idx] = imitation.data.types.TrajectoryWithRew(
                obs = np.concatenate(
                    [demo.obs, np.expand_dims(options[idx], 1)], 1),
                acts = demo.acts,
                infos = demo.infos,
                terminal = demo.terminal,
                rews = demo.rews
            )

    env = gym.make(env_name, **kwargs)
    env = Monitor(env)

    transitions = rollout.flatten_trajectories(rollouts)

    new_logger = set_logger('bc-supervised')
    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=transitions,
        rng=np.random.default_rng(0),
        custom_logger=new_logger,
        device='cpu'
    )
    bc_trainer.train(n_epochs=n_epochs)