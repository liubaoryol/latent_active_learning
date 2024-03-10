import dataclasses
import numpy as np
from typing import List
from abc import ABC, abstractmethod
from .data.types import TrajectoryWithLatent

@dataclasses.dataclass
class Oracle:
    expert_trajectories: List
    true_options: List

    def query(self, trajectory_num, position_num):
        return self.true_options[trajectory_num][position_num]


@dataclasses.dataclass
class CuriousPupil(ABC):
    demos: List[TrajectoryWithLatent]
    oracle: Oracle
    num_queries: int = 0
    @abstractmethod

    def query_oracle(self):
        raise NotImplementedError


@dataclasses.dataclass
class Random(CuriousPupil):
    query_percent: int
    
    def query_oracle(self):
        for idx in range(len(self.demos)):
            self._query_single_demo(idx)
        self.num_queries += 1

    def _query_single_demo(self, idx):
        demo = self.demos[idx]
        for j in range(len(demo)):
            if np.random.uniform() <= self.query_percent:
                option = self.oracle.query(idx, j)
                demo.set_true_latent(j, option)