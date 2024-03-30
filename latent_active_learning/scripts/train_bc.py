import pdb
import os
from datetime import datetime
import numpy as np
import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

from imitation.algorithms import bc
from imitation.data import rollout
from imitation.util import logger as imit_logger
from latent_active_learning.scripts.utils import get_demos, concat_obslat
from latent_active_learning.scripts.config.train_hbc import train_hbc_ex
from latent_active_learning.wrappers.latent_wrapper import FilterLatent
from latent_active_learning.bc import BCLogger
from latent_active_learning.wrappers.latent_wrapper import TransformBoxWorldReward


CURR_DIR = os.getcwd()
timestamp = lambda: datetime.now().strftime('%m-%d-%Y_%H-%M-%S')

def set_logger(
    env,
    wandb_run,
    exp_identifier='behavioral-cloning',
    results_dir='result_fixed_order_targets',
    ):
    boxworld_params = f'size{env.size}-targets{env.n_targets}'

    logging_dir = os.path.join(
        CURR_DIR,
        f'{results_dir}/{boxworld_params}/{exp_identifier}_{timestamp()}/'
        )

    logger = imit_logger.configure(
        logging_dir,
        ["stdout", "csv", "tensorboard"]
        )
    
    return BCLogger(logger, wandb_run)


@train_hbc_ex.automain
def main(_config,
         env_name,
         n_targets,
         filter_state_until,
         kwargs,
         n_epochs,
         use_wandb,
         query_percent):

    if use_wandb:
        import wandb
        run = wandb.init(
            project=f'{env_name[:-3]}-size{kwargs["size"]}-targets{n_targets}',
            name='BehavioralCloning_queryPercent{}_{}'.format(
                query_percent,
                timestamp()
            ),
            tags=['bc'],
            config=_config,
            monitor_gym=True, # NOTE: had to make changes to an __init__ file to make this work. I'm not sure if it will work
            save_code=True
        )
    else:
        run = None

    assert query_percent in [0, 1], '`query_percent` must equal either 0 or 1'

    env = gym.make(env_name, **kwargs)
    env = Monitor(env)
    env.unwrapped._max_episode_steps = kwargs['size']**2
    rollouts, options = get_demos()

    if query_percent:
        env = FilterLatent(env, list(range(filter_state_until, -1)))
        rollouts = concat_obslat(rollouts, options)
    else:
        env = FilterLatent(env, list(range(filter_state_until, 0)))

    transitions = rollout.flatten_trajectories(rollouts)

    logger = set_logger(env, run)

    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=transitions,
        rng=np.random.default_rng(0),
        device='cpu'
    )
    bc_trainer._bc_logger = logger._logger_lo
    for epoch in range(n_epochs):
        mean_return, std_return = evaluate_policy(bc_trainer.policy, TransformBoxWorldReward(env), 10)
        logger.log_batch(
            epoch_num=epoch,
            rollout_mean=mean_return,
            rollout_std=std_return
        )
        bc_trainer.train(n_epochs=1)
