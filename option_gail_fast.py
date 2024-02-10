from imitation.algorithms.adversarial.gail import GAIL
from imitation.rewards.reward_nets import BasicShapedRewardNet, BasicRewardNet
from imitation.util.networks import RunningNorm
from imitation.util import logger as imit_logger
from imitation.policies.base import NormalizeFeaturesExtractor
from torch.nn.modules.activation import ReLU

import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from gymnasium import spaces
from copy import deepcopy as copy
import latent_active_learning
from latent_active_learning.collect import get_expert_trajectories, get_environment

SEED=42
policy_kwargs = {
    "activation_fn": ReLU,
    "features_extractor_class": NormalizeFeaturesExtractor,
    "features_extractor_kwargs": {"normalize_class": RunningNorm},
    "net_arch": [{"pi": [64, 64], "vf": [64, 64]}]
    }

PPO_BoxWorld_Flavor = {
    'batch_size': 128,
    'ent_coef': 0.009709494745755033,
    'gae_lambda': 0.98,
    'gamma': 0.995,
    'learning_rate': 0.0005807211840258373,
    'max_grad_norm': 0.9,
    'n_epochs': 20,
    'vf_coef': 0.20315938606555833,
    'seed': SEED,
    'policy_kwargs': policy_kwargs
}
env_name = "BoxWorld-continuous-v0"
env = gym.make(
    env_name,
    n_targets=2,
    allow_variable_horizon=False
    )

vec_env = get_environment(env_name, False)


ppo = PPO(
    env=vec_env,
    policy=MlpPolicy,
    tensorboard_log="./liu_tensorboard/",
    **PPO_BoxWorld_Flavor
)



reward_net = BasicRewardNet(
    observation_space=vec_env.observation_space,
    action_space=vec_env.action_space,
    normalize_input_layer=RunningNorm,
)

custom_logger = imit_logger.configure(
        folder="liu_tensorboard/airl",
        format_strs=["tensorboard", "stdout"],
)

rollouts = get_expert_trajectories(env_name, 24, False, SEED)

demo_batch_size=32
il = GAIL(
    demonstrations=rollouts,
    demo_batch_size=demo_batch_size,
    gen_replay_buffer_capacity=512,
    n_disc_updates_per_round=16,
    gen_train_timesteps=16,
    venv=vec_env,
    gen_algo=ppo,
    reward_net=reward_net,
    log_dir="./liu_tensorboard/",
    allow_variable_horizon=False, #True,
    init_tensorboard=True,
    init_tensorboard_graph=True,
    custom_logger=custom_logger
    )


class OptionGAIL:
    def __init__(
        self,
        rollouts,
        vec_env,
        gen_algo,
        reward_net
        shared=True,
        dim_c=2
    ):
    self.dim_c = dim_c
    self.env = vec_env
    # Change vec_env observation and action space
    # to fit high and low level policies.
    # Hence, needing to rewrite `collect_rollouts`,
    # as it's done a bit differently.
    # Only will work with Box obs_space

    obs_space = vec_env.observation_space
    new_lo = np.concatenate([obs_space.low, [0]])
    new_hi = np.concatenate([obs_space.high, [self.dim_c]])
    self.env = vec_env
    vec_env_lo = copy(vec_env)
    vec_env_lo.observation_space = spaces.Box(low=new_lo, high=new_hi)
    self.ppo_lo = PPO(
        env=vec_env_lo,
        policy=MlpPolicy
    )

    vec_env_hi = copy(vec_env)
    vec_env_hi.observation_space = spaces.Box(low=new_lo, high=new_hi)
    vec_env_hi.action_space = spaces.Discrete(self.dim_c)
    self.ppo_hi = PPO(
        env=vec_env_hi,
        policy=MlpPolicy
    )

    reward_net_lo = BasicRewardNet(
        observation_space=vec_env_lo.observation_space,
        action_space=vec_env_lo.action_space,
        normalize_input_layer=RunningNorm,
    )

    reward_net_hi = BasicRewardNet(
        observation_space=vec_env_hi.observation_space,
        action_space=vec_env_hi.action_space,
        normalize_input_layer=RunningNorm,
    )

    # Here I am taking advantage of train_disc and train_gen
    self.il_high = GAIL(
        demonstrations=rollouts,
        demo_batch_size=16,
        gen_replay_buffer_capacity=512,
        n_disc_updates_per_round=16,
        gen_train_timesteps=16,
        venv=vec_env_hi,
        gen_algo=self.ppo_hi,
        reward_net=reward_net_hi,
        log_dir="./liu_tensorboard/",
        allow_variable_horizon=False, #True,
        init_tensorboard=True,
        init_tensorboard_graph=True,
        custom_logger=custom_logger
    )
    self.il_low = GAIL(
        demonstrations=rollouts,
        demo_batch_size=16,
        gen_replay_buffer_capacity=512,
        n_disc_updates_per_round=16,
        gen_train_timesteps=16,
        venv=vec_env_lo,
        gen_algo=self.ppo_lo,
        reward_net=reward_net_lo,
        log_dir="./liu_tensorboard/",
        allow_variable_horizon=False, #True,
        init_tensorboard=True,
        init_tensorboard_graph=True,
        custom_logger=custom_logger
    )

    self._last_obs = vec_env.reset()
    self._last_c1 = np.empty([vec_env.num_envs, 1])
    self._last_c1.fill(self.dim_c)
    self._last_episode_starts = np.ones((vec_env.num_envs,), dtype=bool)

    def train(self, num_iterations):
        from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer

        # E-step. Infer expert options with Viterbi?

        # M-step. Update policy low and high'
        # collect_rollouts

        # TODO: you have to do this before collecting rollouts
        total_timesteps, callback = self.ppo_hi._setup_learn(16)
        total_timesteps, callback = self.ppo_lo._setup_learn(16)
        collect_rollouts(
            self.env,
            callback,
            self.ppo_lo.rollout_buffer,
            self.ppo_hi.rollout_buffer,
            n_rollout_steps=10
            )
        
        for r in tqdm.tqdm(range(0, 10), desc="round"):
            self.ppo_hi.train()
            self.il_high.train_gen(self.gen_train_timesteps)
            for _ in range(self.n_disc_updates_per_round):
                with networks.training(self.reward_train):
                    # switch to training mode (affects dropout, normalization)
                    self.il_high.train_disc()
    

        # Train generator with ppo.train() not ppo.learn!
        # Train discriminator
        # Train generator o the other policy
        # Train discriminator of the other policy.
        # For now I will update \pi = \pi_L*\pi_H as a shared network  with updated option
        # M step how long?
        self.policy.train()


    def collect_rollouts(
        # self,
        env, #: VecEnv,
        callback, #: BaseCallback,
        rollout_buffer_lo, #: RolloutBuffer,
        rollout_buffer_hi, #: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Code borrowed from stable_baselines3
        Collect experiences using the current policies (Low and High) and fill
        a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.ppo_lo.policy.set_training_mode(False)
        self.ppo_hi.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer_hi.reset()
        rollout_buffer_lo.reset()

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:

            with torch.no_grad():
                # Convert to pytorch tensor or to TensorDict
                # last obs = obsrevation, which will be input to policy low and hi
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                c1_tensor = obs_as_tensor(self._last_c1, self.device)
                hi_input = torch.cat([obs_tensor, c1_tensor], dim=1)
                # From vec_envs
                c_tensors, c_values, c_log_probs = self.ppo_hi.policy(hi_input)
                c_tensors = c_tensors.reshape(-1, 1)
                lo_input = torch.cat([obs_tensor, c_tensors], dim=1)
                actions, values, log_probs = self.ppo_lo.policy(lo_input)
            c_tensors = c_tensors.cpu().numpy()
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(env.action_space, spaces.Box):
                clipped_actions = np.clip(actions, env.action_space.low, env.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            # TODO: double checek callbacks and this function's arguments
            self.ppo_hi._update_info_buffer(infos)
            self.ppo_lo._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.ppo_lo.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with torch.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                    rewards[idx] += self.gamma * terminal_value

            # How does rewards ar used here? I don't think they should, as those are real?
            # Check what I am saving in buffer
            
            rollout_buffer_hi.add(
                hi_input,  # type: ignore[arg-type]
                c_tensors,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                c_values,
                c_log_probs,
            )
            rollout_buffer_lo.add(
                lo_input,  # type: ignore[arg-type]
                actions,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs,
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_c1 = c_tensors
            self._last_episode_starts = dones

        with torch.no_grad():
            # Compute value for the last timestep
            values_lo = self.ppo_lo.policy.predict_values(obs_as_tensor(np.concatenate([new_obs, c_tensors], axis=1), self.device))  # type: ignore[arg-type]
            values_hi = self.ppo_hi.policy.predict_values(torch.concat([obs_as_tensor(new_obs, self.device), c1_tensor], dim=1))

        rollout_buffer_lo.compute_returns_and_advantage(last_values=values_lo, dones=dones)
        rollout_buffer_hi.compute_returns_and_advantage(last_values=values_hi, dones=dones)
        callback.on_rollout_end()

        return True
    
    # def viterbi_path(self, s_array, a_array):
    #     with torch.no_grad():
    #         log_pis = self.log_prob_action(
    #             s_array,
    #             None,
    #             a_array
    #             ).view(-1, 1, self.dim_c)  # demo_len x 1 x ct
    #         log_trs = self.log_trans(s_array, None)  # demo_len x (ct_1+1) x ct
    #         log_prob = log_trs[:, :-1] + log_pis
    #         log_prob0 = log_trs[0, -1] + log_pis[0, 0]
            
    #         # forward
    #         max_path = torch.empty(len(s_array), c_dim, dtype=torch.long, device=device)
    #         accumulate_logp = log_prob0
    #         max_path[0] = self.dim_c
    #         for i in range(1, s_array.size(0)):
    #             accumulate_logp, max_path[i, :] = (accumulate_logp.unsqueeze(dim=-1) + log_prob[i]).max(dim=-2)
    #         # backward
    #         c_array = torch.zeros( len(s_array)+1, 1, dtype=torch.long, device=device )
    #         log_prob_traj, c_array[-1] = accumulate_logp.max(dim=-1)
    #         for i in range(s_array.size(0), 0, -1):
    #             c_array[i-1] = max_path[i-1][c_array[i]]
    #     return c_array.detach(), log_prob_traj.detach()

    # def log_prob_action(s_array, c_array, a_array):
    #     self.policy_low(s_array, c_array)



rollouts = get_expert_trajectories('BoxWorld-v0', 24, True, SEED)
# Change rollouts to hold only observable states, and latent states have it in separate.
rollouts, latent = separate_obs_latent(rollouts)

def separate_obs_latent(rollouts):
    rollouts_filtered = []
    options = []
    for rollout in rollouts:
        filtered = imitation.data.types.TrajectoryWithRew(
            obs = rollout.obs[:,:-1],
            acts = rollout.acts,
            infos = rollout.infos,
            terminal = rollout.terminal,
            rews = rollout.rews
        )
        rollouts_filtered.append(filtered)
        options.append(rollout.obs[:,-1])
    return rollouts_filtered, options