from typing import List
from .types import TrajectoryWithLatent
from imitation.data.types import TrajectoryWithRew

def augmentTrajectoryWithRew(trajectories: List[TrajectoryWithRew]):
    """Get list with augmented trajectories with latent variables"""

    opts_trajs = []
    for traj in trajectories:
        tmp = TrajectoryWithLatent(
            obs=traj.obs,
            acts=traj.acts,
            infos=traj.infos,
            terminal=traj.terminal
        )
        opts_trajs.append(tmp)

    return opts_trajs
