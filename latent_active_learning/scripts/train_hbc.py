import gymnasium as gym

from latent_active_learning.scripts.config.train_hbc import train_hbc_ex
from latent_active_learning.collect import get_trajectories
from latent_active_learning.collect import filter_TrajsWRewards
from latent_active_learning.collect import get_expert
from latent_active_learning.wrappers.latent_wrapper import FilterLatent
from latent_active_learning.hbc import HBC
from latent_active_learning.oracle import Random, Oracle


@train_hbc_ex.capture
def get_demos(env_name, filter_state_until, kwargs):
    '''Get clean rollouts and corresponding of simple BoxWorld'''
    expert = get_expert(env_name, kwargs)

    rollouts_full = get_trajectories(
        env_name=env_name,
        model=expert,
        kwargs=kwargs,
        n_demo=500
        )

    rollouts = filter_TrajsWRewards(rollouts_full, filter_state_until)
    options = [roll.obs[:,-1] for roll in rollouts_full]

    return rollouts, options


@train_hbc_ex.automain
def main(n_targets,
         filter_state_until,
         query_percent,
         kwargs,
         n_epochs):
    rollouts, options = get_demos()
    gini = Oracle(rollouts, options)
    student = Random(rollouts, gini, option_dim=n_targets, query_percent=query_percent)
    
    student.query_oracle()

    env = gym.make("BoxWorld-v0", **kwargs)
    hbc = HBC(
        option_dim=n_targets,
        device='cpu',
        env=FilterLatent(env, list(range(filter_state_until, 0))),
        exp_identifier=str(query_percent) + 'query_ratio',
        curious_student=student,
        results_dir='results_fixed_order_targets'
        )
    hbc.train(n_epochs)