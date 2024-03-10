import os
from datetime import datetime
import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy

from imitation.algorithms import bc
from imitation.util import logger as imit_logger
from imitation.data import rollout

from latent_active_learning.collect import get_expert_trajectories
from latent_active_learning.collect import get_environment


def get_env_rollouts(env_name, full_obs, size, n_targets, fixed_targets):
    kwargs = {
        'size': size,
        'n_targets': n_targets,
        'allow_variable_horizon': True,
        'fixed_targets': fixed_targets
    }
    # Here assume full observability.
    rollouts = get_expert_trajectories(env_name=env_name,
                                       full_obs=full_obs,
                                       kwargs=kwargs,
                                       n_demo=500)
    transitions = rollout.flatten_trajectories(rollouts)
    env = get_environment(env_name=env_name,
                        full_obs=full_obs,
                        n_envs=1,
                        kwargs=kwargs)
    print(
        f"""The `rollout` function generated a list of {len(rollouts)} {type(rollouts[0])}.
    After flattening, this list is turned into a {type(transitions)} object containing {len(transitions)} transitions.
    The transitions object contains arrays for: {', '.join(transitions.__dict__.keys())}."
    """
    )
    return env, rollouts, transitions


def set_logger(exp_identifier):
    CURR_DIR = os.getcwd()
    timestamp = lambda: datetime.now().strftime('%m-%d-%Y_%H-%M-%S')

    logging_dir = os.path.join(
        CURR_DIR,
        f'results/{exp_identifier}_{timestamp()}/'
        )
    logger = imit_logger.configure(
        logging_dir,
        ["stdout", "csv", "tensorboard"]
        )
    return logger


# Supervised case
env_name = "BoxWorld-v0"
kwargs = {
    'size': 5,
    'n_targets': 2,
    # 'allow_variable_horizon': True,
    'fixed_targets': [[0,0],[4,4]]#, [0,4],]# [9,0],[0,9],[9,9]]
    }

env, rollouts, transitions = get_env_rollouts(
    env_name=env_name,
    full_obs=True,
    **kwargs
)

new_logger = set_logger('bc-supervised')
bc_trainer = bc.BC(
    observation_space=env.observation_space,
    action_space=env.action_space,
    demonstrations=transitions,
    rng=np.random.default_rng(0),
    custom_logger=new_logger,
    device='cpu'
)

reward_before_training, std_before_training = evaluate_policy(bc_trainer.policy, env, 10)
bc_trainer.train(n_epochs=1)
reward_after_training, std_after_training = evaluate_policy(bc_trainer.policy, env, 10)
print(f"Reward after training: {reward_after_training}")

# UNSUPERVISED BC

env, rollouts, transitions = get_env_rollouts(
    env_name=env_name,
    full_obs=False,
    **kwargs
)

bc_trainer = bc.BC(
    observation_space=env.observation_space,
    action_space=env.action_space,
    demonstrations=transitions,
    rng=np.random.default_rng(0),
    custom_logger=new_logger,
    device='cpu'
)

reward_before_training, std_before_training = evaluate_policy(bc_trainer.policy, env, 10)
bc_trainer.train(n_epochs=1)
reward_after_training, std_after_training = evaluate_policy(bc_trainer.policy, env, 10)
print(f"Reward after training: {reward_after_training}")

# CHECK HOW BC WORKS WITH THE OPTIONS ESTIMATED BY AN UNINITIALIZED HBC
from latent_active_learning.hbc import HBC
from latent_active_learning.wrappers.latent_wrapper import FilterLatent


hbc = HBC(
    rollouts,
    options=None,
    option_dim=2,
    device='cpu',
    env=env.envs[0],
    exp_identifier='Nothing',
    query_percent=0
    )

options = hbc.viterbi_list(rollouts)