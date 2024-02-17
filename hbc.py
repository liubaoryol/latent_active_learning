import torch
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from datetime import datetime
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import obs_as_tensor
from typing import List, Union, Tuple, Dict, Optional

import imitation
from imitation.data.types import TrajectoryWithRew
from imitation.algorithms import bc
from imitation.data import rollout
from imitation.util.util import make_vec_env
from imitation.util import logger as imit_logger

import latent_active_learning
from latent_active_learning.collect import get_expert_trajectories
from latent_active_learning.collect import get_environment
from latent_active_learning.collect import filter_intent_TrajsWRewards


CURR_DIR = "/home/liubove/Documents/my-packages/latent_active_learning/"
timestamp = lambda: datetime.now().strftime('%m-%d-%Y_%H-%M-%S')


class HBC:
    def __init__(self,
                 expert_demos: List[TrajectoryWithRew],
                 options: List,
                 option_dim: int,
                 device: str,
                 vec_env,
                 exp_identifier='hbc',
                 query_percent=1
                 ):

        self.expert_demos = expert_demos
        self.options = options
        self.device = device
        self.option_dim = option_dim
        self.query_percent = query_percent

        logging_dir = os.path.join(
            CURR_DIR,
            f'results/{exp_identifier}_{timestamp()}/'
            )

        new_logger = imit_logger.configure(logging_dir,
                                           ["stdout", "csv", "tensorboard"]
                                           )

        obs_space = vec_env.observation_space
        new_lo = np.concatenate([obs_space.low, [0]])
        new_hi = np.concatenate([obs_space.high, [option_dim]])
        action_dim = 4 # TODO: get this from gym environment
        rng = np.random.default_rng(0)

        self.policy_lo = bc.BC(
            observation_space=spaces.Box(low=new_lo, high=new_hi),
            action_space=vec_env.action_space, # Check as sometimes it's continuosu
            rng=rng,
            device=device,
            custom_logger=new_logger
        )
        new_lo[-1] = -1
        self.policy_hi = bc.BC(
            observation_space=spaces.Box(low=new_lo, high=new_hi),
            action_space=spaces.Discrete(option_dim),
            rng=rng,
            device=device,
            # optimizer_cls=torch.optim.SGD,
            # optimizer_kwargs={'lr': 0.01}
        )
        self._logger = self.policy_lo._logger
    

    def train(self, n_epochs):
        N = len(self.expert_demos)
        queries = []
        for i in range(N):
            M = len(self.options[i])
            query = -np.ones(M+1)
            for j in range(M):
                if np.random.uniform() <= self.query_percent:
                    query[j+1] = self.options[i][j]
            queries.append(query)

        for epoch in range(n_epochs):
            # options1 = [np.insert(o, 0, -1).reshape(-1, 1) for o in self.options]
            options = self.viterbi_list(self.expert_demos, queries)
            f = lambda x: np.linalg.norm(
                (options[x].squeeze()[1:] - self.options[x][:-1]), 1)/len(self.options[x])
            distances = list(map(f, range(len(options))))
            self._logger.record("hbc/0-1distance", np.mean(distances))
            self._logger.record("hbc/mean_return", evaluate_policy(hbc, env, 10)[0])
            self._logger.record("hbc/std_return", evaluate_policy(hbc, env, 10)[1])
            
            transitions_lo, transitions_hi = self.get_h_transitions(self.expert_demos, options)
            self.policy_lo.set_demonstrations(transitions_lo)
            self.policy_lo.train(n_epochs=1)
            for j in range(10):
                options = self.viterbi_list(self.expert_demos, queries)
                transitions_lo, transitions_hi = self.get_h_transitions(self.expert_demos, options)
                self.policy_hi.set_demonstrations(transitions_hi)
                self.policy_hi.train(n_epochs=1)

    def get_h_transitions(self, expert_demos, options):
        expert_lo = []
        expert_hi = []
        for idx, demo in enumerate(expert_demos):
            # opts = np.concatenate([[-1], options[idx]]).reshape(-1,1)
            # opts[1:]
            opts = options[idx]
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


    def viterbi_list(self, expert_demos, queries=None):
        options = []
        for demo, query in zip(expert_demos, queries):
            options.append(self.viterbi(demo, query)[0])
        return options

    def viterbi(self, expert_demos, query=None):
        states = expert_demos.obs
        acts = expert_demos.acts

        N = states.shape[0]-1

        if query is None:
            query = -np.ones(N+1, dtype=int)

        with torch.no_grad():
            log_acts = self.log_prob_action(states, acts)  # demo_len x 1 x ct
            log_opts = self.log_prob_option(states)  # demo_len x (ct_1+1) x ct
            # Special handling of last state:
            # log_acts = torch.concatenate([log_acts, torch.zeros([1, self.option_dim])])
            log_acts = log_acts.reshape([-1, 1, self.option_dim])
            log_opts = log_opts[:-1]

            # Done special handling
            log_prob = log_opts[:, 1:] + log_acts
            # log_prob = log_opts[:, :-1] + log_acts
            # log_prob0 = log_opts[0, -1] + log_acts[0, 0]
            # forward
            max_path = torch.empty(N, self.option_dim, dtype=torch.long, device=self.device)
            accumulate_logp = torch.zeros(self.option_dim) #if query[1]>=0 else log_prob0
            # max_path[0] = -1
            for i in range(N):
                if query[i]>=0:
                    accumulate_logp, max_path[i, :] = accumulate_logp + torch.zeros([self.option_dim]), query[i] * torch.ones([self.option_dim])
                else:
                    accumulate_logp, max_path[i, :] = (accumulate_logp.unsqueeze(dim=-1) + log_prob[i]).max(dim=-2)
            # backward
            c_array = -torch.ones(N+1, 1, dtype=torch.long, device=self.device)
            # log_prob_traj, c_array[-1] = accumulate_logp.max(dim=-1)
            log_prob_traj, idx = accumulate_logp.max(dim=-1)
            c_array[-1] = max_path[-1][idx]
            for i in range(N, 1, -1):
                c_array[i-1] = max_path[i-1][c_array[i]]
        return c_array.detach().numpy(), log_prob_traj.detach()
        # return self.options

    def log_prob_option(
        self,
        states: np.ndarray,
        options: Optional[np.ndarray]=None,
        options_1: Optional[np.ndarray]=None):

        states = obs_as_tensor(states, self.device)

        N = states.size(0)
        results = []
        for o in range(-1, self.option_dim):
            input_o = torch.concat([states, torch.ones([N,1])*o], axis=1)
            log_prob=self.policy_hi.policy.get_distribution(input_o).distribution.logits.detach()
            results.append(log_prob)
        return torch.stack(results, axis=1)
    
    def log_prob_action(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        options: Optional[np.ndarray]=None,
        ) -> np.ndarray:
        """
        Return probability P(a|s, o) N x option_dim
        """
        states = obs_as_tensor(states[:-1], self.device)
        actions = obs_as_tensor(actions, self.device)

        N = states.size(0)
        results = []
        if options is not None:
            options = obs_as_tensor(options, self.device)
            input_o = torch.concat([states, options], axis=1)
            return self.policy_lo.policy.get_distribution(input_o).log_prob(actions)

        for o in range(self.option_dim):
            input_o = torch.concat([states, torch.ones([N,1])*o], axis=1)
            log_prob=self.policy_lo.policy.get_distribution(input_o).log_prob(actions)
            results.append(log_prob)
        
        return torch.stack(results, axis=1)

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



# env_name = "BoxWorld-v0"
# kwargs = {
#     'size': 5,
#     'n_targets': 2,
#     'allow_variable_horizon': True,
#     'fixed_targets': [[0,0],[4,4]]
#     }
# from latent_active_learning.collect import train_expert
# try:
#     train_expert(env_name, kwargs)
# except AssertionError:
#     pass
# # Only used for creating the networks of HBC
# env = get_environment(env_name=env_name,
#                       full_obs=False,
#                       n_envs=1,
#                       kwargs=kwargs)

# # Here assume full observability.
# rollouts2 = get_expert_trajectories(env_name=env_name,
#                                     full_obs=True,
#                                     kwargs=kwargs,
#                                     n_demo=500) #[[0,0], [4,4]])
# rollouts = filter_intent_TrajsWRewards(rollouts2)
# options = [rollout.obs[:,-1] for rollout in rollouts2]

# hbc = HBC(rollouts, options, 2, 'cpu', env, exp_identifier='hbc-50%query-diff-approach')

# env = get_environment(env_name=env_name,
#                       full_obs=True,
#                       n_envs=1,
#                       kwargs=kwargs)
# reward_before, std_before = evaluate_policy(hbc, env, 1, render=True)
# hbc.train(30)
# print("Reward:", reward)



# Let's see how hi level policy is learning when we have a trained low level policy


env_name = "BoxWorld-v0"
kwargs = {
    'size': 5,
    'n_targets': 2,
    'allow_variable_horizon': True,
    'fixed_targets': [[0,0],[4,4]]
    }
from latent_active_learning.collect import train_expert
try:
    train_expert(env_name, kwargs)
except AssertionError:
    pass
# Only used for creating the networks of HBC
env = get_environment(env_name=env_name,
                      full_obs=False,
                      n_envs=1,
                      kwargs=kwargs)

# Here assume full observability.
rollouts2 = get_expert_trajectories(env_name=env_name,
                                    full_obs=True,
                                    kwargs=kwargs,
                                    n_demo=500) #[[0,0], [4,4]])
rollouts = filter_intent_TrajsWRewards(rollouts2)
options = [rollout.obs[:,-1] for rollout in rollouts2]

hbc = HBC(
    rollouts,
    options,
    option_dim=2,
    device='cpu',
    vec_env=env,
    exp_identifier='hbc-1-baseline',
    query_percent=1
    )
hbc.train(30)

