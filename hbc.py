import numpy as np
import gymnasium as gym
from stable_baselines3.common.evaluation import evaluate_policy
from gymnasium import spaces
from typing import List, Union, Tuple, Dict, Optional
import torch
import imitation
import dataclasses
import os
from datetime import datetime

import latent_active_learning
from latent_active_learning.collect import get_expert_trajectories, get_environment, filter_intent_TrajsWRewards
from imitation.data.types import Transitions, TrajectoryWithRew
from imitation.algorithms import bc
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies.serialize import load_policy
from imitation.util.util import make_vec_env
from stable_baselines3.common.utils import obs_as_tensor
from imitation.util import logger as imit_logger

CURR_DIR = "/home/liubove/Documents/my-packages/latent_active_learning/"
timestamp = lambda: datetime.now().strftime('%m-%d-%Y_%H-%M-%S')

class HBC:
    def __init__(self,
                 expert_demos: List[TrajectoryWithRew],
                 options: List,
                 option_dim,
                 device,
                 vec_env
                 ):
        self.expert_demos = expert_demos
        self.options = options
        self.device = device
        self.option_dim = option_dim
        action_dim = 4
        rng = np.random.default_rng(0)

        logging_dir = os.path.join(CURR_DIR, f'results/hbc_{timestamp()}/')
        new_logger = imit_logger.configure(logging_dir, ["stdout", "csv", "tensorboard"])

        obs_space = vec_env.observation_space
        new_lo = np.concatenate([obs_space.low, [0]])
        new_hi = np.concatenate([obs_space.high, [option_dim]])
        self.policy_lo = bc.BC(
            observation_space=spaces.Box(low=new_lo, high=new_hi),
            action_space=vec_env.action_space, # Check as sometimes it's continuosu
            rng=rng,
            device=device,
            custom_logger=new_logger
        )
        self.policy_hi = bc.BC(
            observation_space=spaces.Box(low=new_lo, high=new_hi),
            action_space=spaces.Discrete(option_dim),
            rng=rng,
            device=device
        )
        self._logger = self.policy_lo._logger
    
    def train(self, n_epochs):
        for epoch in range(n_epochs):
            # options = [np.insert(o, 0, -1).reshape(-1, 1) for o in self.options]
            options = self.viterbi_list(self.expert_demos)
            transitions_lo, transitions_hi = self.get_h_transitions(self.expert_demos, options)
            self.policy_lo.set_demonstrations(transitions_lo)
            self.policy_hi.set_demonstrations(transitions_hi)
            self.policy_lo.train(n_epochs=30)
            self.policy_hi.train(n_epochs=30)

            f = lambda x: np.linalg.norm(
                (options[x].squeeze()[1:] - self.options[x]), 1)/len(self.options[x])
            distances = list(map(f, range(len(options))))
            self._logger.record("hbc/outer_epoch", epoch)
            self._logger.record("hbc/0-1distance", np.mean(distances))
            self._logger.record("hbc/mean_return", evaluate_policy(hbc, env, 10)[0])
            self._logger.record("hbc/std_return", evaluate_policy(hbc, env, 10)[1])
            
            np.linalg.norm((options[0].squeeze()[1:] - self.options[0]), 1)    

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
                    acts = opts[1:-1],
                    infos = demo.infos,
                    terminal = demo.terminal,
                    rews = demo.rews
                ))
        transitions_hi = rollout.flatten_trajectories(expert_hi)
        transitions_lo = rollout.flatten_trajectories(expert_lo)

        return transitions_lo, transitions_hi


    def viterbi_list(self, expert_demos):
        options = []
        for demo in expert_demos:
            options.append(self.viterbi(demo)[0])
        return options

    def viterbi(self, expert_demos):
        states = expert_demos.obs
        acts = expert_demos.acts

        N = states.shape[0]
        with torch.no_grad():
            log_acts = self.log_prob_action(states, acts)  # demo_len x 1 x ct
            log_opts = self.log_prob_option(states)  # demo_len x (ct_1+1) x ct
            # Special handling of last state:
            log_acts = torch.concatenate([log_acts, torch.zeros([1, self.option_dim])])
            log_acts = log_acts.reshape([-1, 1, self.option_dim])
            # log_opts = log_opts[:-1]

            # Done special handling
            log_prob = log_opts[:, :-1] + log_acts
            log_prob0 = log_opts[0, -1] + log_acts[0, 0]
            # forward
            max_path = torch.empty(N, self.option_dim, dtype=torch.long, device=self.device)
            accumulate_logp = log_prob0
            max_path[0] = -1
            for i in range(1, N):
                accumulate_logp, max_path[i, :] = (accumulate_logp.unsqueeze(dim=-1) + log_prob[i]).max(dim=-2)
            # backward
            c_array = torch.zeros(N+1, 1, dtype=torch.long, device=self.device)
            log_prob_traj, c_array[-1] = accumulate_logp.max(dim=-1)
            for i in range(N, 0, -1):
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



env_name = "BoxWorld-v0"
env = get_environment(env_name, False)
rollouts2 = get_expert_trajectories(env_name, 500, True)
rollouts = filter_intent_TrajsWRewards(rollouts2)
options = [rollout.obs[:,-1] for rollout in rollouts2]

hbc = HBC(rollouts, options, 2, 'cpu', env)
# reward_before, std_before = evaluate_policy(hbc, env, 10)
# hbc.train(200)
# print("Reward:", reward)
