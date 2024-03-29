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

CURR_DIR = os.getcwd()
timestamp = lambda: datetime.now().strftime('%m-%d-%Y_%H-%M-%S')


class HBCLoggerPartial(bc.BCLogger):
    def __init__(self, logger, lo: bool, wandb_run=None):
        super().__init__(logger)
        self.part = 'lo' if lo else 'hi'
        self.wandb_run = wandb_run

    def log_batch(
        self,
        batch_num: int,
        batch_size: int,
        num_samples_so_far: int,
        training_metrics,
        rollout_stats,
    ):
        for k, v in training_metrics.__dict__.items():
            name = f"bc_{self.part}/{k}"
            value = float(v) if v is not None else None
            self._logger.record(name, value)
            if self.wandb_run is not None:
                self.wandb_run.log({name: value})

class HBCLogger:
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
        self._logger_hi = HBCLoggerPartial(logger,
                                           lo=False,
                                           wandb_run=wandb_run)
        self.wandb_run = wandb_run

    def reset_tensorboard_steps(self):
        self._tensorboard_step = 0

    def log_batch(
        self,
        epoch_num: int,
        hamming_loss: int,
        rollout_mean: int,
        rollout_std: int
    ):
        self._logger.record("epoch", epoch_num)
        self._logger.record("env/hamming_loss", hamming_loss)
        self._logger.record("env/rollout_mean", rollout_mean)
        self._logger.record("env/rollout_std", rollout_std)

        self._logger.dump(self._tensorboard_step)
        self._tensorboard_step += 1

        if self.wandb_run is not None:
            self.wandb_run.log({
                "epoch": epoch_num,
                "env/hamming_loss": hamming_loss,
                "env/rollout_mean": rollout_mean,
                "env/rollout_std": rollout_std
            })

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["_logger"]
        return state


class HBC:
    def __init__(self,
                 option_dim: int,
                 device: str,
                 env,
                 exp_identifier='hbc',
                 curious_student=None,
                 results_dir='results',
                 wandb_run=None
                 ):
        self.device = device
        self.option_dim = option_dim
        self.curious_student = curious_student
        self.env = Monitor(env)
        boxworld_params = f'size{env.size}-targets{env.n_targets}'
        logging_dir = os.path.join(
            CURR_DIR,
            f'{results_dir}/{boxworld_params}/{exp_identifier}_{timestamp()}/'
            )

        new_logger = imit_logger.configure(logging_dir,
                                           ["stdout", "csv", "tensorboard"]
                                           )

        self._logger = HBCLogger(new_logger, wandb_run)

        obs_space = env.observation_space
        new_lo = np.concatenate([obs_space.low, [0]])
        new_hi = np.concatenate([obs_space.high, [option_dim]])
        rng = np.random.default_rng(0)

        self.policy_lo = bc.BC(
            observation_space=spaces.Box(low=new_lo, high=new_hi),
            action_space=env.action_space, # Check as sometimes it's continuosu
            rng=rng,
            device=device
        )
        self.policy_lo._bc_logger = self._logger._logger_lo
        new_lo[-1] = -1
        self.policy_hi = bc.BC(
            observation_space=spaces.Box(low=new_lo, high=new_hi),
            action_space=spaces.Discrete(option_dim),
            rng=rng,
            device=device
        )
        self.policy_hi._bc_logger = self._logger._logger_hi
        TrajectoryWithLatent.set_policy(self.policy_lo, self.policy_hi)


    def train(self, n_epochs):
        self.curious_student.query_oracle()

        for epoch in range(n_epochs):
            with torch.no_grad():
                options = [demo.latent for demo in self.curious_student.demos]
                f = lambda x: np.linalg.norm(
                    (options[x].squeeze()[1:] - self.curious_student.oracle.true_options[x]), 0)/len(options[x])
                distances = list(map(f, range(len(options))))
            mean_return, std_return = evaluate_policy(self, TransformBoxWorldReward(self.env), 10)
            self._logger.log_batch(
                epoch_num=epoch,
                hamming_loss=np.mean(distances),
                rollout_mean=mean_return,
                rollout_std=std_return
            )

            self._logger.last_mean = mean_return
            self._logger.last_std = std_return
            self._logger.last_01_distance = np.mean(distances)
            
            transitions_lo, transitions_hi = self.transitions(self.curious_student.demos)
            self.policy_lo.set_demonstrations(transitions_lo)
            self.policy_lo.train(n_epochs=1)
            self.policy_hi.set_demonstrations(transitions_hi)
            self.policy_hi.train(n_epochs=1)

    def transitions(self, expert_demos):
        expert_lo = []
        expert_hi = []
        for demo in expert_demos:
            opts = demo.latent
            expert_lo.append(imitation.data.types.TrajectoryWithRew(
                obs = np.concatenate([demo.obs, opts[1:]], axis=1),
                acts = demo.acts,
                infos =  demo.infos,
                terminal = demo.terminal,
                rews = demo.rews
            )
            )
            expert_hi.append(
                imitation.data.types.TrajectoryWithRew(
                    obs = np.concatenate([demo.obs, opts[:-1]], axis=1),
                    acts = opts[1:-1].reshape(-1),
                    infos = demo.infos,
                    terminal = demo.terminal,
                    rews = demo.rews
                ))
        transitions_hi = rollout.flatten_trajectories(expert_hi)
        transitions_lo = rollout.flatten_trajectories(expert_lo)

        return transitions_lo, transitions_hi

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ):
        if state is None:
            n = len(observation)
            state = -np.ones(n)
        state[episode_start] = -1

        hi_input = obs_as_tensor(np.concatenate([observation, state.reshape(-1, 1)], axis=1), device=self.device)
        state, _ = self.policy_hi.policy.predict(hi_input)
        lo_input = obs_as_tensor(np.concatenate([observation, state.reshape(-1, 1)], axis=1), device=self.device)
        actions, _ = self.policy_lo.policy.predict(lo_input)
        return actions, state

    def save(self):
        policy_lo_state = self.policy_lo.policy.state_dict()
        policy_hi_state = self.policy_hi.policy.state_dict()
        path = os.path.join(self._logger.dir, 'model_params.tar')
        torch.save({
            'policy_lo_state': policy_lo_state,
            'policy_hi_state': policy_hi_state
            }, path)

    def load(self, path):
        model_state_dict = torch.load(path)
        self.policy_lo.policy.load_state_dict(model_state_dict['policy_lo_state'])
        self.policy_hi.policy.load_state_dict(model_state_dict['policy_hi_state'])

    def save_config(self):
        # TODO: Finish function.
        save_dict = {
            'size': self.env.size,
            'n_targets': self.env.n_targets,
            'allow_variable_horizon': self.allow_variable_horizon,
            'fixed_targets': self.env.fixed_targets
        }
        # TODO: Save query type and query params.
