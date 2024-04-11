from datetime import datetime
import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
import wandb

from latent_active_learning.scripts.config.train_hbc import train_hbc_ex
from latent_active_learning.scripts.utils import get_demos
from latent_active_learning.wrappers.latent_wrapper import FilterLatent
from latent_active_learning.hbc import HBC
from latent_active_learning.oracle import Random, Oracle, QueryCapLimit, EfficientStudent
from latent_active_learning.oracle import *
from latent_active_learning.collect import get_dir_name

timestamp = lambda: datetime.now().strftime('%m-%d-%Y_%H-%M-%S')

@train_hbc_ex.automain
def main(_config,
         env_name,
         n_targets,
         filter_state_until,
         kwargs,
         n_epochs,
         use_wandb,
         student_type=None,
         efficient_student=False,
         query_percent=None,
         query_cap=None,
         exp_identifier=None):
    
    # error_msg = 'Must define one and only one of `num_queries` or `query_percent`'
    # assert query_percent is None or query_cap is None, error_msg
    # assert not (query_percent is None and query_cap is None), error_msg

    if use_wandb:
        run = wandb.init(
            project=f'[{exp_identifier}]{env_name[:-3]}-size{kwargs["size"]}-targets{n_targets}',
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

    path = get_dir_name(env_name, kwargs).split('/')[1]
    gini = Oracle.load('./expert_trajs/{}'.format(path))
    rollouts = gini.expert_trajectories

    # Create student:
    if student_type=='random':
        student = Random(rollouts, gini, option_dim=n_targets, query_percent=query_percent)

    elif student_type=='query_cap':
        student = QueryCapLimit(rollouts, gini, option_dim=n_targets, query_demo_cap=query_cap)

    elif student_type=='iterative_random':
        student = IterativeRandom(rollouts, gini, option_dim=n_targets)

    elif student_type=='action_entropy':
        student = ActionEntropyBased(rollouts, gini, option_dim=n_targets)

    elif student_type=='intent_entropy':
        student = IntentEntropyBased(rollouts, gini, option_dim=n_targets)

    elif student_type=='tamada':
        student = Tamada(rollouts, gini, option_dim=n_targets)
    
    elif student_type=='action_intent_entropy':
        student = ActionIntentEntropyBased(rollouts, gini, option_dim=n_targets)


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
    hbc.train(n_epochs)

    print(f'Student queried {hbc.curious_student._num_queries} amount of times')