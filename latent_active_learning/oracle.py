import dataclasses
import numpy as np
from typing import List
from abc import ABC, abstractmethod
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
        """Will query oracle on all trajectories and states at the rate of `query_percent`"""
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
