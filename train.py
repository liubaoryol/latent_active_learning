from latent_active_learning.hbc import HBC
import argparse
import json
import numpy as np
import time
import os
import shutil
# import psutil
import sys
import socket
import traceback

from collections import OrderedDict

import torch
from torch.utils.data import DataLoader

import robomimic
import robomimic.utils.train_utils as TrainUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
from robomimic.config import config_factory
from robomimic.algo import algo_factory, RolloutPolicy
from robomimic.utils.log_utils import PrintLogger, DataLogger, flush_warnings
from types import SimpleNamespace   
args = {
    'config': "/home/liubove/Documents/git-packages/robomimic/robomimic/exps/paper/core/square/mh/low_dim/hbc.json"
    }
args = SimpleNamespace(**args)


ext_cfg = json.load(open(args.config, 'r'))
config = config_factory(ext_cfg["algo_name"])
with config.values_unlocked():
    config.update(ext_cfg)
# first set seeds
np.random.seed(config.train.seed)
torch.manual_seed(config.train.seed)
torch.set_num_threads(2)

# Set directories and config
log_dir, ckpt_dir, video_dir = TrainUtils.get_exp_dir(config)
ObsUtils.initialize_obs_utils_with_config(config)

# make sure the dataset exists
dataset_path = os.path.expanduser(config.train.data)
env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=config.train.data)
shape_meta = FileUtils.get_shape_metadata_from_dataset(
    dataset_path=config.train.data,
    all_obs_keys=config.all_obs_keys,
    verbose=True
)

# create environment
envs = OrderedDict()
# create environments for validation runs
env_names = [env_meta["env_name"]]

for env_name in env_names:
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        env_name=env_name, 
        render=False, 
        render_offscreen=config.experiment.render_video,
        use_image_obs=shape_meta["use_images"],
        # use_depth_obs=shape_meta["use_depths"],
    )
    env = EnvUtils.wrap_env_from_config(env, config=config) # apply environment warpper, if applicable
    envs[env.name] = env

device='cpu'
num_workers=0
# Set Dataset and model
# setup for a new training run
data_logger = DataLogger(
    log_dir,
    config,
    log_tb=config.experiment.logging.log_tb,
    log_wandb=config.experiment.logging.log_wandb,
)
model = algo_factory(
    algo_name=config.algo_name,
    config=config,
    obs_key_shapes=shape_meta["all_shapes"],
    ac_dim=shape_meta["ac_dim"],
    device=device,
)
trainset, validset = TrainUtils.load_data_for_training(
    config, obs_keys=shape_meta["all_obs_keys"])
train_sampler = trainset.get_dataset_sampler()
valid_sampler = validset.get_dataset_sampler()
# initialize data loaders
train_loader = DataLoader(
    dataset=trainset,
    sampler=train_sampler,
    batch_size=config.train.batch_size,
    shuffle=(train_sampler is None),
    num_workers=config.train.num_data_workers,
    drop_last=True
)
valid_loader = DataLoader(
    dataset=validset,
    sampler=valid_sampler,
    batch_size=config.train.batch_size,
    shuffle=(valid_sampler is None),
    num_workers=num_workers,
    drop_last=True
)


# main training loop
best_valid_loss = None
best_return = {k: -np.inf for k in envs} if config.experiment.rollout.enabled else None
best_success_rate = {k: -1. for k in envs} if config.experiment.rollout.enabled else None
last_ckpt_time = time.time()

# number of learning steps per epoch (defaults to a full dataset pass)
train_num_steps = config.experiment.epoch_every_n_steps
valid_num_steps = config.experiment.validation_epoch_every_n_steps


for epoch in range(1, config.train.num_epochs + 1): # epoch numbers start at 1
    step_log = TrainUtils.run_epoch(
        model=model,
        data_loader=train_loader,
        epoch=epoch,
        num_steps=train_num_steps,
        obs_normalization_stats=None,
    )
    model.on_epoch_end(epoch)





batch = next(iter(train_loader))
input_batch = model.process_batch_for_training(batch)


























env_name = "BoxWorld-v0"
env = get_environment(env_name, False)
rollouts2 = get_expert_trajectories(env_name, 500, True, SEED)
rollouts = filter_intent_TrajsWRewards(rollouts2)
options = [rollout.obs[:,-1] for rollout in rollouts2]

hbc = HBC(rollouts, options, 2, 'cpu', env)
reward_before, std_before = evaluate_policy(hbc, env, 10)
hbc.train(200)
print("Reward:", reward)