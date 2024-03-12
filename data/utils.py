from typing import List
from .types import TrajectoryWithLatent
from imitation.data.types import TrajectoryWithRew

def augmentTrajectoryWithLatent(trajectories: List[TrajectoryWithRew], option_dim):
    """Get list with augmented trajectories with latent variables"""

    opts_trajs = []
    for traj in trajectories:
        tmp = TrajectoryWithLatent(
            obs=traj.obs,
            acts=traj.acts,
            infos=traj.infos,
            terminal=traj.terminal,
            rews=traj.rews
        )
        opts_trajs.append(tmp)

    TrajectoryWithLatent.set_option_dim(option_dim)
    return opts_trajs
