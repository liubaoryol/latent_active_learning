#!/usr/bin/env python3

import os
import torch
import numpy as np
import pandas as pd
import gymnasium as gym
from typing import Union
import matplotlib.pyplot as plt

from latent_active_learning.wrappers.latent_wrapper import FilterLatent
from latent_active_learning.baselines.OptionGAIL.model.option_ppo import OptionPPO
from latent_active_learning.baselines.OptionGAIL.model.option_gail import OptionGAIL
from latent_active_learning.baselines.OptionGAIL.utils.utils import validate, reward_validate, get_dirs, set_seed
from latent_active_learning.baselines.OptionGAIL.utils.agent import Sampler
from latent_active_learning.baselines.OptionGAIL.utils.logger import Logger
from latent_active_learning.baselines.OptionGAIL.utils.config import ARGConfig
from latent_active_learning.baselines.OptionGAIL.default_config import mujoco_config as config
from latent_active_learning.collect import get_expert_trajectories, get_environment
from latent_active_learning.envs.boxworld import trajs2demo


def make_gail(config, dim_s, dim_a):
    gail = OptionGAIL(config, dim_s=dim_s, dim_a=dim_a)
    ppo = OptionPPO(config, gail.policy)
    return gail, ppo


def train_g(ppo: OptionPPO, sample_sxar, factor_lr):
    ppo.step(sample_sxar, lr_mult=factor_lr)


def train_d(gail: OptionGAIL, sample_sxar, demo_sxar, n_step=10):
    return gail.step(sample_sxar, demo_sxar, n_step=n_step)


def sample_batch(gail: OptionGAIL, agent, n_sample, demo_sa_array):
    sample_sxar_in = agent.collect(gail.policy.state_dict(), n_sample, fixed=False)
    # Change rewards to the rewards outputed by gail
    sample_sxar, sample_rsum = gail.convert_sample(sample_sxar_in)
    demo_sxar, demo_rsum = gail.convert_demo(demo_sa_array)
    return sample_sxar, demo_sxar, sample_rsum, demo_rsum


def log(il, sampling_agent, demo_sxar, iteration, rewards_df, logger, sample_r, demo_r, config, msg='default'):
    v_l, cs_demo = validate(il.policy, [(tr[0], tr[-2]) for tr in demo_sxar])
    logger.log_test("expert_logp", v_l, iteration)
    info_dict, cs_sample = reward_validate(sampling_agent, il.policy, do_print=True)
    logger.log_test_info(info_dict, iteration)
    tmp_dict = {'Algorithm': [config.algo],
                'seed': [config.seed],
                'env': [config.env_name + '-fullyObs-' + str(config.full_obs)],
                'iteration': [iteration],
                'performance': [info_dict['r-avg']]}

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

    # # Set all directories and files for logging
    log_dir, save_dir, sample_name, pretrain_name = get_dirs(config.seed,
                                                             config.algo,
                                                             env_type,
                                                             env_name,
                                                             msg,
                                                             config.use_option
                                                             )
    with open(os.path.join(save_dir, "config.log"), 'w') as f:
        f.write(str(config))
    logger = Logger(log_dir)
    save_name_f = lambda i: os.path.join(save_dir, f"gail_{i}.torch")
    rewards_df = pd.DataFrame(columns=['Algorithm', 'seed', 'env',
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


    # Create learning networks
    gail, ppo = make_gail(config, dim_s=dim_s, dim_a=dim_a)
    sampling_agent = Sampler(config.seed, env, gail.policy)

    # Process data
    demo = trajs2demo(rollouts)
    demo_sa_array = tuple((s.to(gail.device), a.to(gail.device)) for s, a, r in demo)
    sample_sxar, demo_sxar, sample_r, demo_r = sample_batch(gail, sampling_agent, n_sample, demo_sa_array)
    
    # Initial logging
    rewards_df = log(gail,
                     sampling_agent,
                     demo_sxar,
                     0,
                     rewards_df,
                     logger,
                     sample_r,
                     demo_r,
                     config
                     )


    for i in range(n_epoch):
        print(i)
        sample_sxar, demo_sxar, sample_r, demo_r = sample_batch(gail, sampling_agent, n_sample, demo_sa_array)

        train_d(gail, sample_sxar, demo_sxar, n_step=5)
        train_g(ppo, sample_sxar, factor_lr=1.)
        if (i + 1) % config.log_interval == 0:

            rewards_df = log(gail,
                             sampling_agent,
                             demo_sxar,
                             i,
                             rewards_df,
                             logger,
                             sample_r,
                             demo_r,
                             config
                             )
            # if (i + 1) % (100) == 0:
            #     torch.save((gail.state_dict(), sampling_agent.state_dict()), save_name_f(i))

        logger.log_train("r-sample-avg", sample_r, i)
        logger.log_train("r-demo-avg", demo_r, i)
        logger.flush()

    return rewards_df

if __name__ == "__main__":
    arg = ARGConfig()
    arg.add_arg("use_option", True, "Use Option when training or not")
    arg.add_arg("env_name", "BoxWorld-v0", "Environment name")
    arg.add_arg("device", "cpu", "Computing device")
    arg.add_arg("tag", "default", "Experiment tag")
    arg.add_arg("n_demo", 30, "Number of demonstration s-a")
    arg.add_arg("n_epoch", 2000, "Number of training epochs")
    arg.add_arg("seed", torch.randint(100, ()).item(), "Random seed")
    arg.add_arg("use_c_in_discriminator", True, "Use (s,a) or (s,c,a) as occupancy measurement")
    arg.add_arg("use_d_info_gail", False, "Use directed-info gail or not")
    arg.add_arg("train_option", True, "Train master policy or not (only false when using D-info-GAIL)")
    arg.add_arg("full_obs", False, "Full observability of BoxWorld or without intent")

    arg.parser()

    config.update(arg)
    config.algo = 'option-gail'
    config.discrete = True if config.env_name=='BoxWorld-v0' else False

    config.shared_policy = True
    print(f">>>> Training {'Option-' if config.use_option else ''}GAIL using {config.env_name} environment on {config.device}")

    # list_rewards = []
    # for seed in tqdm(range(5)):
    #     config.seed = seed
    rewards_df = learn(config, msg=config.tag)
    #     list_rewards.append(rewards_df)