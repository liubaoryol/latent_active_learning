from datetime import datetime
import gymnasium as gym
from stable_baselines3.common.monitor import Monitor

from latent_active_learning.scripts.config.train_hbc import train_hbc_ex
from latent_active_learning.scripts.utils import get_demos
from latent_active_learning.wrappers.latent_wrapper import FilterLatent
from latent_active_learning.hbc import HBC
from latent_active_learning.oracle import Random, Oracle, QueryCapLimit


timestamp = lambda: datetime.now().strftime('%m-%d-%Y_%H-%M-%S')

@train_hbc_ex.automain
def main(_config,
         env_name,
         n_targets,
         filter_state_until,
         kwargs,
         n_epochs,
         use_wandb,
         query_percent=None,
         query_cap=None):
    
    error_msg = 'Must define one and only one of `num_queries` or `query_percent`'
    assert query_percent is None or query_cap is None, error_msg
    assert not (query_percent is None and query_cap is None), error_msg

    if use_wandb:
        import wandb
        run = wandb.init(
            project=f'{env_name[:-3]}-size{kwargs["size"]}-targets{n_targets}',
            name='HBC_{}{}_{}'.format(
                'queryCap' if query_cap is not None else 'queryPercent',
                query_cap if query_cap is not None else query_percent,
                timestamp()
            ),
            tags=['hbc'],
            config=_config,
            monitor_gym=True, # NOTE: had to make changes to an __init__ file to make this work. I'm not sure if it will work
            save_code=True
        )
    else:
        run = None

    rollouts, options = get_demos()
    gini = Oracle(rollouts, options)
    if query_percent is not None:
        student = Random(rollouts, gini, option_dim=n_targets, query_percent=query_percent)
    else:
        student = QueryCapLimit(rollouts, gini, option_dim=n_targets, query_demo_cap=query_cap)

    student.query_oracle()


    env = gym.make(env_name, **kwargs)
    env = Monitor(env)
    env = FilterLatent(env, list(range(filter_state_until, 0)))
    # env = DummcyVecEnv([])
    # env = VecVideoRecorder(
    #     env,
    #     f'videos/{run.id if run is not None else _config.seed}', 
    #     record_video_trigger=lambda x: x % 2000==0,
    #     video_length=200
    # )

    hbc = HBC(
        option_dim=n_targets,
        device='cpu',
        env=env,
        exp_identifier=str(query_percent) + 'query_ratio',
        curious_student=student,
        results_dir='results_fixed_order_targets',
        wandb_run=run
        )
    hbc.train(n_epochs)