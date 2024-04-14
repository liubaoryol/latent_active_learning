import shutil
import os
import wandb
import gymnasium as gym
from datetime import datetime
from stable_baselines3.common.monitor import Monitor

from latent_active_learning.scripts.config.train_hbc import train_hbc_ex
from latent_active_learning.scripts.utils import get_demos
from latent_active_learning.collect import get_dir_name
from latent_active_learning.wrappers.latent_wrapper import FilterLatent
from latent_active_learning.hbc import HBC
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
         filter_state_until,
         kwargs,
         use_wandb,
         student_type=None,
         efficient_student=False,
         query_percent=None,
         query_cap=None,
         exp_identifier='',
         n_epochs=None):

    path = get_dir_name(env_name, kwargs).split('/')[1]
    gini = Oracle.load('./expert_trajs/{}'.format(path))
    rollouts = gini.expert_trajectories

    if use_wandb:
        run = wandb.init(
            project=f'{exp_identifier}{path}',
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
    env = FilterLatent(env, list(range(filter_state_until, 0)))
    env.unwrapped._max_episode_steps = kwargs['size']**2 * n_targets

    hbc = HBC(
        option_dim=n_targets,
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