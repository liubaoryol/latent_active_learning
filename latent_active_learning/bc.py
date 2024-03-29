import torch
import os
from datetime import datetime
from typing import Union, Tuple, Dict, Optional
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import obs_as_tensor

import imitation
from imitation.algorithms import bc
from imitation.data import rollout
from imitation.util import logger as imit_logger

from latent_active_learning.data.types import TrajectoryWithLatent
from latent_active_learning.wrappers.latent_wrapper import TransformBoxWorldReward
from latent_active_learning.hbc import HBCLoggerPartial

CURR_DIR = os.getcwd()
timestamp = lambda: datetime.now().strftime('%m-%d-%Y_%H-%M-%S')


class BCLogger:
    """Utility class to help logging information relevant to Behavior Cloning."""

    def __init__(self, logger: imit_logger.HierarchicalLogger, wandb_run=None):
        """Create new BC logger.

        Args:
            logger: The logger to feed all the information to.
        """
        self._logger = logger
        self._tensorboard_step = 0
        self._current_epoch = 0
        self._logger_lo = HBCLoggerPartial(logger,
                                           lo=True,
                                           wandb_run=wandb_run)
        self.wandb_run = wandb_run

    def reset_tensorboard_steps(self):
        self._tensorboard_step = 0

    def log_batch(
        self,
        epoch_num: int,
        rollout_mean: int,
        rollout_std: int
    ):
        self._logger.record("epoch", epoch_num)
        self._logger.record("env/rollout_mean", rollout_mean)
        self._logger.record("env/rollout_std", rollout_std)

        self._logger.dump(self._tensorboard_step)
        self._tensorboard_step += 1

        if self.wandb_run is not None:
            self.wandb_run.log({
                "epoch": epoch_num,
                "env/rollout_mean": rollout_mean,
                "env/rollout_std": rollout_std
            })

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["_logger"]
        return state
