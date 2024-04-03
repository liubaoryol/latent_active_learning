import logging
import dataclasses
import numpy as np
from typing import List
from abc import ABC, abstractmethod
from stable_baselines3.common.utils import obs_as_tensor

from imitation.data.types import TrajectoryWithRew
from .data.types import TrajectoryWithLatent
from .data.utils import augmentTrajectoryWithLatent


@dataclasses.dataclass
class Oracle:
    expert_trajectories: List[TrajectoryWithRew]
    true_options: List[np.ndarray]

    def query(self, trajectory_num, position_num):
        return self.true_options[trajectory_num][position_num]
    
    def __str__(self):
        return f'Oracle(num_demos={len(self.expert_trajectories)})'


@dataclasses.dataclass
class CuriousPupil(ABC):
    demos: List[TrajectoryWithLatent]
    oracle: Oracle
    option_dim: int
    
    def __post_init__(self):
        self._num_queries = 0
        self.demos = augmentTrajectoryWithLatent(self.demos, self.option_dim)

    @abstractmethod
    def query_oracle(self):
        raise NotImplementedError

    def __str__(self):
        return f'Student(num_demos={len(self.demos)}, option_dim={self.option_dim})'
    

@dataclasses.dataclass
class Random(CuriousPupil):
    query_percent: int = 0
    
    def query_oracle(self):
        """Will query oracle on all trajectories and states at the rate of `query_percent`"""
        for idx in range(len(self.demos)):
            self._query_single_demo(idx)
        self._num_queries += 1

    def _query_single_demo(self, idx):
        demo = self.demos[idx]
        for j in range(len(demo.obs)):
            if np.random.uniform() <= self.query_percent:
                option = self.oracle.query(idx, j)
                demo.set_true_latent(j, option)

@dataclasses.dataclass
class QueryCapLimit(CuriousPupil):
    query_demo_cap: int = 0
    
    def query_oracle(self):
        """Will query oracle on all trajectories `query_demo_cap` number of times"""
        for idx in range(len(self.demos)):
            self._query_single_demo(idx)
        self._num_queries += 1

    def _query_single_demo(self, idx):
        demo = self.demos[idx]
        n = len(demo.obs)
        idxs = np.random.choice(range(n), size=min(self.query_demo_cap, n))
        for j in idxs:
            option = self.oracle.query(idx, j)
            demo.set_true_latent(j, option)


@dataclasses.dataclass
class EfficientStudent(CuriousPupil):
    """Student that accesses all info, but stores only the
    states at the change of the latent state"""
    
    def query_oracle(self):
        for idx in range(len(self.demos)):
            self._query_single_demo(idx)
        self._num_queries += 1

    def _query_single_demo(self, idx):
        demo = self.demos[idx]

        option_1 = self.oracle.query(idx, 0)
        demo.set_true_latent(0, option_1)
        for j in range(1, len(demo.obs)):
            option = self.oracle.query(idx, j)
            if option!=option_1:
                demo.set_true_latent(j, option)
                option_1=option


@dataclasses.dataclass
class IterativeRandom(CuriousPupil):
    def query_oracle(self):
        """Will query oracle on all trajectories and states, randomly"""
        for idx in range(len(self.demos)):
            self._query_single_demo(idx)
        self._num_queries += 1

    def _query_single_demo(self, idx):
        # Query intent at a random timestep of the demo
        demo = self.demos[idx]
        n = list(range(len(demo.obs)))
        unlabeled_idxs = np.array(n)[~demo._is_latent_estimated[1:]]
        if unlabeled_idxs.size > 0:
            j = np.random.choice(unlabeled_idxs)
            option = self.oracle.query(idx, j)
            demo.set_true_latent(j, option)
        else:
            logging.warn("All latent states in demo have been queried")


@dataclasses.dataclass
class ActionEntropyBased(CuriousPupil):
    policy: object = None

    def set_policy(self, model):
        self.policy = model.policy_lo.policy

    def query_oracle(self):
        """Will query oracle on all trajectories and states, randomly"""
        # TODO: most probably it is better to have a
        # buffer thats quashes all the demos into one
        for idx in range(len(self.demos)):
            self._query_single_demo(idx)
        self._num_queries += 1

    def _query_single_demo(self, idx):
        # Get options, which are calculated using Viterbi
        demo = self.demos[idx]
        n = list(range(len(demo.obs)))
        unlabeled_idxs = np.array(n)[~demo._is_latent_estimated[1:]]

        if unlabeled_idxs.size > 0
            observations = demo.obs[unlabeled_idxs]
            options = demo.latent[unlabeled_idxs+1]
            
            with torch.no_grad():
                lo_input = obs_as_tensor(
                    np.concatenate([observations, options], axis=1),
                    device=self.model.device)
                entropy = self.policy.get_distribution(lo_input).entropy()
                top_entropy_idx = entropy.topk(1)[1].item()

            top_entropy_idx = unlabeled_idxs[top_entropy_idx]
            option = self.oracle.query(idx, top_entropy_idx)
            demo.set_true_latent(top_entropy_idx, option)

@dataclasses.dataclass
class IntentEntropyBased(CuriousPupil):
    def query_oracle(self):
        """Will query oracle on all trajectories and states, randomly"""
        for idx in range(len(self.demos)):
            self._query_single_demo(idx)
        self._num_queries += 1

    def _query_single_demo(self, idx):
        pass


@dataclasses.dataclass
class ActionIntentEntropyBased(CuriousPupil):
    def query_oracle(self):
        """Will query oracle on all trajectories and states, randomly"""
        for idx in range(len(self.demos)):
            self._query_single_demo(idx)
        self._num_queries += 1

    def _query_single_demo(self, idx):
        pass
