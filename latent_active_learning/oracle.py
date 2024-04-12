import os
import pickle
import logging
import dataclasses
import numpy as np
from typing import List
from abc import ABC, abstractmethod
from stable_baselines3.common.utils import obs_as_tensor

from imitation.data.types import TrajectoryWithRew
from .data.types import TrajectoryWithLatent
from .data.utils import augmentTrajectoryWithLatent


@dataclasses.dataclass(frozen=True)
class QueryIdentifier:
    traj_num: int
    idx_query: int

@dataclasses.dataclass
class Oracle:
    expert_trajectories: List[TrajectoryWithRew]
    true_options: List[np.ndarray]
    expert_trajectories_test: List[TrajectoryWithRew] = None
    true_options_test: List[np.ndarray] = None

    def query(self, trajectory_num, position_num):
        return self.true_options[trajectory_num][position_num]
    
    def __str__(self):
        return f'Oracle(num_demos={len(self.expert_trajectories)})'
    
    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        for name, attribute in self.__dict__.items():
            name += '.pkl'
            with open("/".join((path, name)), "wb") as f:
                pickle.dump(attribute, f)

    @property
    def max_queries(self) -> int:
        num_queries = 0
        for demo in self.expert_trajectories:
            num_queries+=len(demo.obs)
        return num_queries

    @classmethod
    def load(cls, path):
        my_model = {}
        for name in cls.__annotations__:
            
            with open("/".join((path, name+'.pkl')), "rb") as f:
                my_model[name] = pickle.load(f)
        return cls(**my_model)

    def __eq__(self, other):        
        A = other.expert_trajectories == self.expert_trajectories
        B = other.expert_trajectories_test == self.expert_trajectories_test
        C = np.all([(opts == other_opts).all() for opts, other_opts
                    in zip(self.true_options, other.true_options)])
        if self.true_options_test is None:
            D = other.true_options_test is None
        else:
            D = np.all([(opts == other_opts).all() for opts, other_opts 
                        in zip(self.true_options_test, other.true_options_test)])
        return A and B and C and D

    def stats(self):
        obs_len = [len(tr.obs) for tr in self.expert_trajectories]
        returns = [np.sum(tr.rews) for tr in self.expert_trajectories]
        obs_t_len = [len(tr.obs) for tr in self.expert_trajectories_test]
        returns_t = [np.sum(tr.rews) for tr in self.expert_trajectories_test]
        print("STATS TRAINING DATASET")
        print("Num trajectories: ", len(self.expert_trajectories))
        print("Avereage length: {}+/-{}".format(
            np.mean(obs_len),
            np.std(obs_len)))
        print("Average returns: {}+/-{}\n".format(
            np.mean(returns),
            np.std(returns)
        ))

        print("STATS TESTING DATASET")
        print("Num trajectories: ", len(self.expert_trajectories))
        print("Avereage length: {}+/-{}".format(
            np.mean(obs_t_len),
            np.std(obs_t_len)))
        print("Average returns: {}+/-{}".format(
            np.mean(returns_t),
            np.std(returns_t)
        ))

@dataclasses.dataclass
class CuriousPupil(ABC):
    oracle: Oracle
    option_dim: int
    
    def __post_init__(self):
        self._num_queries = 0
        self.list_queries = []
        self.demos = augmentTrajectoryWithLatent(
            self.oracle.expert_trajectories,
            self.option_dim
        )
        self.demos_test = augmentTrajectoryWithLatent(
            self.oracle.expert_trajectories_test,
            self.option_dim
        )
    @abstractmethod
    def query_oracle(self, num_queries):
        raise NotImplementedError

    def save_query(self, traj_num, idx_query):
        self.list_queries.append(QueryIdentifier(traj_num, idx_query))

    def __str__(self):
        return f'Student(num_demos={len(self.demos)}, option_dim={self.option_dim})'

    def save_queries(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        name = 'list_queries.pkl'
        attribute = self.__dict__['list_queries']
        with open("/".join((path, name)), "wb") as f:
            pickle.dump(attribute, f)

@dataclasses.dataclass
class Random(CuriousPupil):
    query_percent: int = 0
    student_type: str = f'query_percent{query_percent}'

    def query_oracle(self, num_queries=1):
        """Will query oracle on all trajectories and states at the rate of `query_percent`"""
        if self._num_queries > 0:
            return 
        for idx in range(len(self.demos)):
            self._query_single_demo(idx)
        self._num_queries += 1

    def _query_single_demo(self, idx):
        demo = self.demos[idx]
        for j in range(len(demo.obs)):
            if np.random.uniform() <= self.query_percent:
                option = self.oracle.query(idx, j)
                demo.set_true_latent(j, option)
                self.save_query(idx, j)

@dataclasses.dataclass
class QueryCapLimit(CuriousPupil):
    query_demo_cap: int = 0
    student_type: str = f'query_cap{query_demo_cap}'
    
    def query_oracle(self, num_queries=1):
        """Will query oracle on all trajectories `query_demo_cap` number of times"""
        if self._num_queries > 0:
            return 
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
            self.save_query(idx, j)


@dataclasses.dataclass
class EfficientStudent(CuriousPupil):
    """Student that accesses all info, but stores only the
    states at the change of the latent state"""
    student_type: str = 'efficient'

    def query_oracle(self, num_queries=1):
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
                self.save_query(idx, j)


@dataclasses.dataclass
class IterativeRandom(CuriousPupil):
    student_type: str = 'iterative_random'
    def query_oracle(self, num_queries=1):
        for _ in range(num_queries):
            self._query_oracle()

    def _query_oracle(self):
        """Will query oracle on all trajectories and states, randomly"""
        idx = np.random.randint(len(self.demos))
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
            self.save_query(idx, j)
        else:
            logging.warn("All latent states in demo have been queried")


@dataclasses.dataclass
class ActionEntropyBased(CuriousPupil):
    policy: object = None
    student_type: str = 'action_entropy'

    def set_policy(self, model):
        self.policy = model.policy_lo.policy

    def query_oracle(self, num_queries=1):
        for _ in range(num_queries):
            self._query_oracle()
        
    def _query_oracle(self):
        """Will query oracle on all trajectories and states, randomly"""
        # TODO: most probably it is better to have a
        # buffer thats quashes all the demos into one
        top_entropies = []
        top_entropies_idx = []
        for idx in range(len(self.demos)):
            ent_idx, ent = self._get_info_single_demo(idx)
            top_entropies.append(ent)
            top_entropies_idx.append(ent_idx)

        idx_traj = np.argmax(top_entropies)
        top_entropy_idx = top_entropies_idx[idx_traj]
        if top_entropy_idx is not None:
            option = self.oracle.query(idx_traj, top_entropy_idx)
            demo = self.demos[idx_traj]
            demo.set_true_latent(top_entropy_idx, option)
            self.save_query(idx_traj, top_entropy_idx)
            self._num_queries += 1

    def _get_info_single_demo(self, idx):
        # Get options, which are calculated using Viterbi
        demo = self.demos[idx]
        n = list(range(len(demo.obs)))
        unlabeled_idxs = np.array(n)[~demo._is_latent_estimated[1:]]
        top_entropy_idx = None
        top_entropy = 0
        if unlabeled_idxs.size > 0:
            observations = demo.obs[unlabeled_idxs]
            options = demo.latent[unlabeled_idxs+1]
            import torch
            with torch.no_grad():
                lo_input = obs_as_tensor(
                    np.concatenate([observations, options], axis=1),
                    device=self.policy.device)
                entropy = self.policy.get_distribution(lo_input).entropy()
                top_entropy, top_entropy_idx = entropy.topk(1)
                top_entropy = top_entropy.item()
                top_entropy_idx = top_entropy_idx.item()

            top_entropy_idx = unlabeled_idxs[top_entropy_idx]
    
        return top_entropy_idx, top_entropy


@dataclasses.dataclass
class IntentEntropyBased(CuriousPupil):
    student_type: str = 'intent_entropy'

    def query_oracle(self, num_queries=1):
        for _ in range(num_queries):
            self._query_oracle()

    def _query_oracle(self):
        """Will query oracle on all trajectories and states, randomly"""
        
        top_entropies = []
        top_entropies_idx = []
        for idx in range(len(self.demos)):
            ent_idx, ent = self._get_info_single_demo(idx)
            top_entropies.append(ent)
            top_entropies_idx.append(ent_idx)
        
        idx_traj = np.argmax(top_entropies)
        top_entropy_idx = top_entropies_idx[idx_traj]
        if top_entropy_idx is not None:
            option = self.oracle.query(idx_traj, top_entropy_idx)
            demo = self.demos[idx_traj]
            demo.set_true_latent(top_entropy_idx, option)
            self.save_query(idx_traj, top_entropy_idx)
            self._num_queries += 1

    def _get_info_single_demo(self, idx):
        demo = self.demos[idx]
        entropy = demo.entropy()
        top_entropy_idx = entropy[1:].argmax()
        top_entropy = entropy[1+top_entropy_idx]
        return top_entropy_idx, top_entropy

@dataclasses.dataclass
class ActionIntentEntropyBased(CuriousPupil):
    mixing: float=0.5
    policy: object = None
    student_type: str = 'mixed_action_entropy'

    def set_policy(self, model):
        self.policy = model.policy_lo.policy

    def query_oracle(self, num_queries=1):
        for _ in range(num_queries):
            self._query_oracle()

    def _query_oracle(self):
        """Will query oracle on all trajectories and states, randomly"""
        top_entropies = []
        top_entropies_idx = []

        for idx in range(len(self.demos)):
            ent_idx, ent = self._get_info_single_demo(idx)
            top_entropies.append(ent)
            top_entropies_idx.append(ent_idx)
        
        idx_traj = np.argmax(top_entropies)
        top_entropy_idx = top_entropies_idx[idx_traj]
        
        if top_entropy_idx is not None:
            option = self.oracle.query(idx_traj, top_entropy_idx)
            demo = self.demos[idx_traj]
            demo.set_true_latent(top_entropy_idx, option)
            self.save_query(idx_traj, top_entropy_idx)
            self._num_queries += 1

    def _get_info_single_demo(self, idx):

        demo = self.demos[idx]
        n = list(range(len(demo.obs)))
        entropies = np.zeros(len(demo.obs))
        unlabeled_idxs = np.array(n)[~demo._is_latent_estimated[1:]]

        entropy_intent = demo.entropy()

        if unlabeled_idxs.size > 0:
            observations = demo.obs[unlabeled_idxs]
            options = demo.latent[unlabeled_idxs+1]
            import torch
            with torch.no_grad():
                lo_input = obs_as_tensor(
                    np.concatenate([observations, options], axis=1),
                    device=self.policy.device)
                entropy_action = self.policy.get_distribution(lo_input).entropy()

                entropies[unlabeled_idxs] = entropy_action * self.mixing   
        
        entropies += (1-self.mixing) * entropy_intent[1:]

        top_entropy_idx = entropies.argmax()
        return top_entropy_idx, entropies[top_entropy_idx]


@dataclasses.dataclass
class Unsupervised(Random):
    def __post_init__(self):
        super().__post_init__()
        self.student_type = 'unsupervised'
        self.query_percent = 0

@dataclasses.dataclass
class Supervised(Random):
    def __post_init__(self):
        super().__post_init__()
        self.student_type = 'supervised'
        self.query_percent = 1

