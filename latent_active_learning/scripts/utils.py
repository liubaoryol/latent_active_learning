from latent_active_learning.scripts.config.train_hbc import train_hbc_ex
from latent_active_learning.collect import get_trajectories
from latent_active_learning.collect import filter_TrajsWRewards
from latent_active_learning.collect import get_expert


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