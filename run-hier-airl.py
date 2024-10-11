#!/usr/bin/env python3
import os
import torch
import numpy as np
import pandas as pd
import gymnasium as gym
from typing import Union
import matplotlib.pyplot as plt
from tqdm import tqdm

from latent_active_learning.wrappers.latent_wrapper import FilterLatent
from latent_active_learning.baselines.HierAIRL.HierAIRL_Ant.model.MHA_option_ppo import MHAOptionPPO
from latent_active_learning.baselines.HierAIRL.HierAIRL_Ant.model.MHA_option_il import MHAOptionAIRL, MHAOptionGAIL
from latent_active_learning.baselines.HierAIRL.HierAIRL_Ant.utils.common_utils import validate, reward_validate, get_dirs, set_seed
from latent_active_learning.baselines.HierAIRL.HierAIRL_Ant.sampler import Sampler
from latent_active_learning.baselines.HierAIRL.HierAIRL_Ant.utils.logger import Logger
from latent_active_learning.baselines.HierAIRL.HierAIRL_Ant.utils.config import ARGConfig
from latent_active_learning.baselines.HierAIRL.HierAIRL_Ant.default_config import mujoco_config as config
from latent_active_learning.collect import get_expert_trajectories, get_environment
from latent_active_learning.envs.boxworld import trajs2demo


def make_il(config, dim_s, dim_a):
    if config.is_airl:
        il = MHAOptionAIRL(config, dim_s=dim_s, dim_a=dim_a)
    else:
        il = MHAOptionGAIL(config, dim_s=dim_s, dim_a=dim_a)
    ppo = MHAOptionPPO(config, il.policy)
    return il, ppo


def train_g(ppo: MHAOptionPPO, sample_sxar, factor_lr):
    ppo.step(sample_sxar, lr_mult=factor_lr)


def train_d(il: Union[MHAOptionGAIL, MHAOptionAIRL], sample_sxar, demo_sxar, n_step=10):
    il.step(sample_sxar, demo_sxar, n_step=n_step)


def sample_batch(il: Union[MHAOptionGAIL, MHAOptionAIRL], agent, n_sample, demo_sa_array):
    demo_sa_in = agent.filter_demo(demo_sa_array)
    sample_sxar_in = agent.collect(il.policy.state_dict(), n_sample, fixed=False)
    # replace the real environment reward with the one generated with IL
    sample_sxar, sample_rsum = il.convert_sample(sample_sxar_in)
    # Estimate options and calculate rewards with options
    demo_sxar, demo_rsum = il.convert_demo(demo_sa_in)
    return sample_sxar, demo_sxar, sample_rsum, demo_rsum


def log(il, sampling_agent, demo_sxar, iteration, rewards_df, logger, sample_r, demo_r, msg='default'):
    v_l, cs_demo = validate(il.policy, [(tr[0], tr[-2]) for tr in demo_sxar])
    logger.log_test("expert_logp", v_l, iteration)
    info_dict, cs_sample = reward_validate(sampling_agent, il.policy, do_print=True)
    logger.log_test_info(info_dict, iteration)
    tmp_dict = {'Algorithm': [config.algo],
                'seed': [config.seed],
                'iteration': [info_dict['r-avg']],
                'performance': [iteration]}

    rewards_df = pd.concat([rewards_df, pd.DataFrame(tmp_dict)])
    
    print(f"{iteration}: r-sample-avg={sample_r}, r-demo-avg={demo_r}, log_p={v_l} ; {msg}, performance={info_dict['r-avg']}")
    
    logger.log_train("r-sample-avg", sample_r, iteration)
    logger.log_train("r-demo-avg", demo_r, iteration)
    return rewards_df


def learn(config, msg="default"):
    n_demo = config.n_demo
    n_sample = config.n_sample
    n_thread = config.n_thread
    n_epoch = config.n_epoch
    env_name = config.env_name
    env_type = config.env_type
    set_seed(config.seed)

    # Set logging
    log_dir, save_dir, sample_name, _ = get_dirs(config.seed,
                                                 config.algo,
                                                 env_type,
                                                 env_name,
                                                 msg)
    with open(os.path.join(save_dir, "config.log"), 'w') as f:
        f.write(str(config))  # important for reproducing and visualisation
    logger = Logger(log_dir)  # tensorboard
    save_name_f = lambda i: os.path.join(save_dir, f"{i}.torch")
    rewards_df = pd.DataFrame(columns=['Algorithm', 'seed',
                                       'iteration', 'performance'
                                       ])

    # Create environment and expert data
    env = gym.make(env_name, n_targets=2)
    config.dim_c = env.unwrapped.n_targets
    if not config.full_obs:
        env = FilterLatent(env, unobservable_states=[-1])
    rollouts = get_expert_trajectories(env_name, arg.n_demo, arg.full_obs, arg.seed)

    dim_s = env.observation_space.shape[0]
    if config.discrete:
        dim_a = env.action_space.n
    else:
        dim_a = env.action_space.shape[0]


    # Create learning algorithms
    il, ppo = make_il(config, dim_s=dim_s, dim_a=dim_a)
    sampling_agent = Sampler(config.seed, env, il.policy, n_thread=n_thread)

    # Process data
    demo = trajs2demo(rollouts)
    demo_sa_array = tuple((s.to(il.device), a.to(il.device)) for s, a, r in demo)
    sample_sxar, demo_sxar, sample_r, demo_r = sample_batch(il, sampling_agent, n_sample, demo_sa_array)

    # Initial logging
    rewards_df = log(il,
                     sampling_agent,
                     demo_sxar,
                     0,
                     rewards_df,
                     logger,
                     sample_r,
                     demo_r
                     )

    for i in range(n_epoch):
        print(i)
        sample_sxar, demo_sxar, sample_r, demo_r = sample_batch(il, sampling_agent, n_sample, demo_sa_array)
        if i % 3 == 0:
            train_d(il, sample_sxar, demo_sxar)

        train_g(ppo, sample_sxar, factor_lr=1.)


        if (i + 1) % config.log_interval == 0:

            rewards_df = log(il,
                             sampling_agent,
                             demo_sxar,
                             i,
                             rewards_df,
                             logger,
                             sample_r,
                             demo_r
                             )
            if (i + 1) % (100) == 0:
                torch.save(il.state_dict(), save_name_f(i))
        
        logger.log_train("r-sample-avg", sample_r, i)
        logger.log_train("r-demo-avg", demo_r, i)
        logger.flush()
    return rewards_df


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


if __name__ == '__main__':

    arg = ARGConfig()
    arg.add_arg("env_type", "mujoco", "Environment type, can be [mujoco, rlbench]")
    arg.add_arg("env_name", "BoxWorld-v0", "Environment name")
    arg.add_arg("algo", "hier_airl", "which algorithm to use, can be [option_airl, hier_airl, hier_gail]")
    arg.add_arg("device", "cpu", "Computing device")
    arg.add_arg("tag", "default", "Experiment tag")
    arg.add_arg("seed", 0, "Random seed")
    arg.add_arg("n_demo", 30, "Number of demonstration s-a")
    arg.add_arg("full_obs", False, "Full observability of BoxWorld or without intent")

    arg.parser()

    config.update(arg)
    if 'airl' in config.algo:
        config.is_airl = True
    else:
        config.is_airl = False

    if 'hier' in config.algo:
        config.use_posterior = True
    else:
        config.use_posterior = False

    config.use_c_in_discriminator = True
    config.use_vae = False
    config.discrete = True if config.env_name=='BoxWorld-v0' else False

    print(f">>>> Training {config.algo} using {config.env_name} environment on {config.device}")

    list_rewards = []
    for seed in tqdm(range(5)):
        config.seed = seed
        rewards_df = learn(config, msg=config.tag)
        list_rewards.append(rewards_df)