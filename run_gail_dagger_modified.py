import numpy as np
import torch
from torch.nn.modules.activation import ReLU
import gymnasium as gym
import pandas as pd
from tqdm import tqdm
from gymnasium.spaces import Discrete

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ppo import MlpPolicy
from imitation.algorithms.adversarial.airl import AIRL
from imitation.algorithms.adversarial.dagger_gail import GAIL
from imitation.data import rollout
from imitation.rewards.reward_nets import BasicShapedRewardNet, BasicRewardNet
from imitation.util.networks import RunningNorm
from imitation.policies.base import NormalizeFeaturesExtractor
from imitation.util import logger as imit_logger

from latent_active_learning.baselines.HierAIRL.HierAIRL_Ant.utils.config import ARGConfig
from latent_active_learning.collect import get_expert_trajectories, get_environment, get_expert

import tempfile

from imitation.algorithms import bc
from imitation.algorithms.dagger import SimpleDAggerTrainer, DAggerTrainer, DAggerTrainerGAIL

SEED=42
policy_kwargs = {
    "activation_fn": ReLU,
    "features_extractor_class": NormalizeFeaturesExtractor,
    "features_extractor_kwargs": {"normalize_class": RunningNorm},
    "net_arch": [{"pi": [64, 64], "vf": [64, 64]}]
    }

PPO_Hopper_Flavor = {
    'batch_size': 512,
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


def make_il(algo, expert, env=None, rollouts=None, demo_batch_size=128):

    ppo = PPO(
        env=env,
        policy=MlpPolicy,
        tensorboard_log="./liu_tensorboard/",
        **PPO_BoxWorld_Flavor
    )

    # Reward net model
    if algo=='airl':
        reward_net = BasicShapedRewardNet(
            observation_space=env.observation_space,
            action_space=env.action_space, # Discrete and continuous should be handled
            normalize_input_layer=RunningNorm
        )
    else:
        reward_net = BasicRewardNet(
            observation_space=env.observation_space,
            action_space=env.action_space,
            normalize_input_layer=RunningNorm,
        )
    
    custom_logger = imit_logger.configure(
            folder="liu_tensorboard/airl",
            format_strs=["tensorboard", "stdout"],
    )


    ALGO_CLASS = GAIL if algo=='gail' else AIRL

    il = ALGO_CLASS(
        demonstrations=rollouts,
        expert=expert,
        demo_batch_size=demo_batch_size,
        gen_replay_buffer_capacity=512,
        n_disc_updates_per_round=16,
        gen_train_timesteps=16,
        venv=env,
        gen_algo=ppo,
        reward_net=reward_net,
        log_dir="./liu_tensorboard/",
        allow_variable_horizon=False,
        init_tensorboard=True,
        init_tensorboard_graph=True,
        custom_logger=custom_logger
        )
    return il, ppo


def update_df(args, rewards, iteration, rewards_df):
    tmp_dict = {'Algorithm': [args.algo],
                'seed': [args.seed],
                'env': [args.env_name + '-fullyObs-' + str(args.full_obs)],
                'iteration': [iteration],
                'performance': [np.mean(rewards)]}

    return pd.concat([rewards_df, pd.DataFrame(tmp_dict)])


def appendto_csv(file_name, rewards_df):
    df = pd.read_csv(file_name)
    df = pd.concat([df, rewards_df])
    df.to_csv(file_name, index=False)


if __name__ == "__main__":
    arg = ARGConfig()
    arg.add_arg("env_name", "BoxWorld-v0", "Environment name: BoxWorld-v0, Walker2d-v4, Hopper-v4")
    arg.add_arg("device", "cuda:0", "Computing device")
    arg.add_arg("n_demo", 1, "Number of demonstration s-a")
    arg.add_arg("n_epoch", 800, "Number of training epochs")
    arg.add_arg("seed", torch.randint(100, ()).item(), "Random seed")
    arg.add_arg("algo", "gail", "Imitation Algorithm: option-gail, hier-airl, airl")
    arg.add_arg("full_obs", True, "Full observability of BoxWorld or without intent")
    arg.parser()


    PPO_BoxWorld_Flavor = {
        'batch_size': 16,
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
    

    expert = get_expert(arg.env_name, arg.n_demo, arg.full_obs)


    vec_env = get_environment(arg.env_name, arg.full_obs)
    rollouts = get_expert_trajectories(arg.env_name, arg.n_demo, arg.full_obs, arg.seed)

    # IL algorithm
    il, ppo = make_il(arg.algo, expert, vec_env, rollouts, demo_batch_size=16)
    vec_env.seed(SEED)
    learner_rewards_before_training, _ = evaluate_policy(
        ppo, vec_env, 10, return_episode_rewards=True,
    )

    rewards_df = pd.DataFrame(columns=['Algorithm', 'seed', 'env',
                                    'iteration', 'performance'
                                    ])

    # for step in tqdm(range(int(arg.n_epoch/16))):
    il.train(16 * arg.n_epoch)
    
    with tempfile.TemporaryDirectory(prefix="dagger_example_") as tmpdir:
        print(tmpdir)
        dagger_trainer = DAggerTrainerGAIL(
            venv=vec_env,
            scratch_dir=tmpdir,
            expert_policy=expert,
            bc_trainer=il,
            rng=np.random.default_rng(),
        )

        dagger_trainer.train(
            25*50,
            rollout_round_min_episodes=1,
            rollout_round_min_timesteps=1,
            # num_timesteps_to_update_grail
            )

    # a = il._demo_data_loader.dataset 
    # a = dagger_trainer.bc_trainer._demo_data_loader.data_loader.dataset

        
    for seed in tqdm(range(5)):
        arg.seed = seed
        algo = arg.algo
        env_name = arg.env_name
        # Create parallel environments
        vec_env = get_environment(env_name, arg.full_obs)
        rollouts = get_expert_trajectories(env_name, arg.n_demo, arg.full_obs, arg.seed)

        # IL algorithm
        il, ppo = make_il(algo, vec_env, rollouts)
        vec_env.seed(SEED)
        learner_rewards_before_training, _ = evaluate_policy(
            ppo, vec_env, 10, return_episode_rewards=True,
        )

        rewards_df = pd.DataFrame(columns=['Algorithm', 'seed', 'env',
                                        'iteration', 'performance'
                                        ])

        # for step in tqdm(range(int(arg.n_epoch/16))):
        il.train(16 * arg.n_epoch)
        rewards, _ = evaluate_policy(
            ppo, vec_env, 10, return_episode_rewards=True,
        )
        rewards_df = update_df(arg, rewards, step, rewards_df)

        appendto_csv('result/results', rewards_df)

