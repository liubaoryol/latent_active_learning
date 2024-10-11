import os
import wandb
import gymnasium as gym
from datetime import datetime
from stable_baselines3.common.monitor import Monitor
import numpy as np
import torch
import random

from latent_active_learning.scripts.config.train_hbc import train_hbc_ex
from latent_active_learning.scripts.utils import get_demos
from latent_active_learning.collect import get_dir_name
from latent_active_learning.wrappers.latent_wrapper import FilterLatent
from latent_active_learning.wrappers.movers_wrapper import MoversAdapt, MoversBoxWorldRepr
from latent_active_learning.wrappers.movers_wrapper import MoversFullyObs
from latent_active_learning.hbc import HBC
# from latent_active_learning.hbc_v2 import HBCv2
from latent_active_learning.oracle import Oracle, Random, QueryCapLimit
from latent_active_learning.oracle import IntentEntropyBased
from latent_active_learning.oracle import ActionEntropyBased
from latent_active_learning.oracle import ActionIntentEntropyBased
from latent_active_learning.oracle import EfficientStudent
from latent_active_learning.oracle import Supervised, Unsupervised, IterativeRandom

timestamp = lambda: datetime.now().strftime('%m-%d-%Y_%H-%M-%S')

@train_hbc_ex.automain
def main(_config,
         env_name,
         n_targets,
         filter_state_until=None,
         kwargs={},
         use_wandb=False,
         student_type=None,
         query_percent=None,
         query_cap=None,
         exp_identifier='',
         n_epochs=None,
         movers_optimal=True,
         options_w_robot=True,
         state_w_robot_opts=False,
         fixed_latent=True,
         box_repr=True,
         seed=0):
    """Hierarchical Behavior Cloning
    0. Set up wandb, seeds, path names
    1. Create Oracle and Student
        Oracle holds the expert trajectories and latents
        Student options:
            iterative_random
            supervised
            unsupervised
            intent_entropy
            action_entropy
    2. Create env
        Env is only used for getting observation space and
        action space in the creation of policies
        It is also used for interaction during evaluation
    3. Create HBC
    4. Train
    """
    # Set up
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    if env_name=="TeamBoxWorld-v0":
        path ="/home/liubove/Documents/my-packages/rw4t-dataset/" \
            "dataset/trajectories/discrete/gini_n18"
        gini = Oracle.load(path)

        # possibly need to clean up gini
        for expts in [gini.expert_trajectories, gini.expert_trajectories_test]:
            for expt in expts:
                filter_out = [0,1, 10]
                obs = expt.obs[:, filter_out]
                object.__setattr__(expt, 'obs', obs)

    elif env_name=="rw4t-continuous":
        path ="/home/liubove/Documents/my-packages/rw4t-dataset/" \
            "dataset/trajectories/continuous/gini_n18"
        gini = Oracle.load(path)
        raise NotImplementedError

    if use_wandb:
        run = wandb.init(
            project=f'{exp_identifier}rw4t-discrete',
            name='HBC_{}{}'.format(
                student_type,
                timestamp(),
            ),
            tags=['hbc'],
            config=_config,
            save_code=True
        )
    else:
        run = None

    # Create student:
    if student_type=='random':
        student = Random(gini, option_dim=n_targets, query_percent=query_percent)

    elif student_type=='query_cap':
        student = QueryCapLimit(gini, option_dim=n_targets, query_demo_cap=query_cap)

    elif student_type=='iterative_random':
        student = IterativeRandom(gini, option_dim=n_targets)

    elif student_type=='action_entropy':
        student = ActionEntropyBased(gini, option_dim=n_targets)

    elif student_type=='intent_entropy':
        student = IntentEntropyBased(gini, option_dim=n_targets)

    elif student_type=='supervised':
        student = Supervised(gini, option_dim=n_targets)

    elif student_type=='unsupervised':
        student = Unsupervised(gini, option_dim=n_targets)
    
    elif student_type=='action_intent_entropy':
        student = ActionIntentEntropyBased(gini, option_dim=n_targets)

    env = gym.make(env_name, **kwargs)
    env = Monitor(env)
    # filtered = list(range(filter_state_until, 0))
    filtered = [2,3,4,5,6,7,8, 9, 11]
    
    env = FilterLatent(env, filtered)
    env.unwrapped._max_episode_steps = kwargs['size']**2 /2
    # Create algorithm with student (who has oracle), env
    hbc = HBC(
        option_dim=n_targets, # check how are the categories selected?
        device='cpu',
        env=env,
        exp_identifier=str(query_percent) + 'query_ratio',
        curious_student=student,
        results_dir='results_fixed_order_targets',
        wandb_run=run
        )

    if student_type in ['action_entropy', 'action_intent_entropy']:
        student.set_policy(hbc)

    if n_epochs is None:
        delay_after_query_exhaustion = 50
        n_epochs = gini.max_queries + delay_after_query_exhaustion
    hbc.train(n_epochs)

    # Log table with metrics
    if run:
        run.log({"metrics/metrics": hbc._logger.metrics_table})
        table = hbc._logger.metrics_table.get_dataframe()
        base_path = os.path.join(os.getcwd(), f'csv_metrics/{path}/')
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        file_name = f'{base_path}hbc_{student_type}{timestamp()}.csv'
        table.to_csv(file_name, header=True, index=False)
        wandb.save(file_name, base_path=os.getcwd())

        file_name = file_name.split('.')[0] + '_list_queries.npy'
        hbc.curious_student.save_queries(file_name)
        wandb.save(file_name, base_path=os.getcwd())