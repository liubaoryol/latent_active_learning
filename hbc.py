import torch
import os
import numpy as np
from gymnasium import spaces
from datetime import datetime
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import obs_as_tensor
from typing import List, Union, Tuple, Dict, Optional
from wandb.integration.sb3 import WandbCallback

import imitation
from imitation.algorithms import bc
from imitation.data import rollout
from imitation.util import logger as imit_logger

from latent_active_learning.data.types import TrajectoryWithLatent
from latent_active_learning.wrappers.latent_wrapper import TransformBoxWorldReward

CURR_DIR = os.getcwd()
timestamp = lambda: datetime.now().strftime('%m-%d-%Y_%H-%M-%S')
   

class HBC:
    def __init__(self,
                 option_dim: int,
                 device: str,
                 env,
                 exp_identifier='hbc',
                 curious_student=None,
                 results_dir='results'
                 ):
        self.device = device
        self.option_dim = option_dim
        self.curious_student = curious_student
        self.env = env
        boxworld_params = f'size{env.size}-targets{env.n_targets}'
        logging_dir = os.path.join(
            CURR_DIR,
            f'{results_dir}/{boxworld_params}/{exp_identifier}_{timestamp()}/'
            )

        new_logger = imit_logger.configure(logging_dir,
                                           ["stdout", "csv", "tensorboard"]
                                           )

        obs_space = env.observation_space
        new_lo = np.concatenate([obs_space.low, [0]])
        new_hi = np.concatenate([obs_space.high, [option_dim]])
        rng = np.random.default_rng(0)

        self.policy_lo = bc.BC(
            observation_space=spaces.Box(low=new_lo, high=new_hi),
            action_space=env.action_space, # Check as sometimes it's continuosu
            rng=rng,
            device=device,
            custom_logger=new_logger
        )
        new_lo[-1] = -1
        self.policy_hi = bc.BC(
            observation_space=spaces.Box(low=new_lo, high=new_hi),
            action_space=spaces.Discrete(option_dim),
            rng=rng,
            device=device
        )
        self._logger = self.policy_lo._logger
        TrajectoryWithLatent.set_policy(self.policy_lo, self.policy_hi)


    def train(self, n_epochs):
        self.curious_student.query_oracle()

        for _ in range(n_epochs):
            with torch.no_grad():
                options = [demo.latent for demo in self.curious_student.demos]
                f = lambda x: np.linalg.norm(
                    (options[x].squeeze()[1:] - self.curious_student.oracle.true_options[x]), 0)/len(options[x])
                distances = list(map(f, range(len(options))))

            self._logger.record("hbc/0-1distance", np.mean(distances))
            mean_return, std_return = evaluate_policy(self, TransformBoxWorldReward(self.env), 10)
            self._logger.record("hbc/mean_return", mean_return)
            self._logger.record("hbc/std_return", std_return)
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
