import os
import wandb
import gymnasium as gym
from datetime import datetime
# from gymnasium.wrappers import Monitor
import numpy as np
import torch
import random
import imitation

from latent_active_learning.scripts.config.train_hbc import train_hbc_ex
from latent_active_learning.scripts.utils import get_demos
from latent_active_learning.collect import get_dir_name
from latent_active_learning.wrappers.franka_wrapper import ObservationOnly
from latent_active_learning.hbc import HBC
from latent_active_learning.oracle import Oracle, Random, QueryCapLimit
from latent_active_learning.oracle import IntentEntropyBased
from latent_active_learning.oracle import ActionEntropyBased
from latent_active_learning.oracle import ActionIntentEntropyBased
from latent_active_learning.oracle import EfficientStudent
from latent_active_learning.oracle import Supervised, Unsupervised, IterativeRandom

timestamp = lambda: datetime.now().strftime('%m-%d-%Y_%H-%M-%S')

@train_hbc_ex.automain
def main(_config,
         env_name,
         n_targets,
         filter_state_until=None,
         kwargs={},
         use_wandb=False,
         student_type=None,
         query_percent=None,
         query_cap=None,
         exp_identifier='',
         n_epochs=None,
         movers_optimal=True,
         options_w_robot=True,
         state_w_robot_opts=False,
         fixed_latent=True,
         box_repr=True,
         seed=0):
    """Hierarchical Behavior Cloning
    0. Set up wandb, seeds, path names
    1. Create Oracle and Student
        Oracle holds the expert trajectories and latents
        Student options:
            iterative_random
            supervised
            unsupervised
            intent_entropy
            action_entropy
    2. Create env
        Env is only used for getting observation space and
        action space in the creation of policies
        It is also used for interaction during evaluation
    3. Create HBC
    4. Train
    """
    # Set up
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    if env_name=="FrankaKitchen-v1":
        path_train ="/home/liubove/Documents/my-packages/" \
            "latent_active_learning/expert_trajs/FrankaKitchen-v0-mixed_260_array_train.pkl"
        path_test ="/home/liubove/Documents/my-packages/" \
            "latent_active_learning/expert_trajs/FrankaKitchen-v0-mixed_260_array_test.pkl"
        import minari 
        from imitation.data.types import TrajectoryWithRew

        expert_trajs = []
        true_options = []
        dataset = minari.load_dataset('D4RL/kitchen/complete-v2', download=True)
        
        for episode in dataset:
            obj_state = episode.observations['achieved_goal']
            obj_goal = episode.observations['desired_goal']
            arg1 = np.linalg.norm(obj_state['microwave'] - obj_goal['microwave'], axis=1).argmin()
            arg2 = np.ceil(np.linalg.norm(obj_state['kettle'] - obj_goal['kettle'], axis=1)*100).argmin()
            arg3 = np.ceil(np.linalg.norm(obj_state['light switch'] - obj_goal['light switch'], axis=1)*100).argmin()

            obs = episode.observations['observation'][:arg1]
            acts = episode.actions[:arg1-1]
            rews = episode.rewards[:arg1-1]
            trj = TrajectoryWithRew(
                obs = obs,
                acts = acts,
                infos = None,
                terminal = True,
                rews = rews
            )
            expert_trajs.append(trj)

            opts = np.zeros(len(rews) + 1)
            opts[arg1:arg2] = 1
            opts[arg2:arg3] = 2
            opts[arg3:] = 3
            true_options.append(opts[:arg1])


        # trajs = np.load(path_train, allow_pickle=True)
        # expert_trajs = []
        # true_options = []
        # for st, acts, rews, dones, latents in trajs:
        #     trj = imitation.data.types.TrajectoryWithRew(
        #         obs = st,
        #         acts = acts,
        #         infos = None,
        #         terminal = dones[-1],
        #         rews = rews[:-1]
        #     )
        #     expert_trajs.append(trj)
        #     latents[latents==4] = 3
        #     true_options.append(latents)


        # expert_trajs_test = []
        # true_options_test = []
        # trajs = np.load(path_test, allow_pickle=True)
        # for st, acts, rews, dones, latents in trajs:
        #     trj = imitation.data.types.TrajectoryWithRew(
        #         obs = st,
        #         acts = acts,
        #         infos = None,
        #         terminal = dones[-1],
        #         rews = rews[:-1]
        #     )
        #     expert_trajs_test.append(trj)
        #     latents[latents==4] = 3
        #     true_options_test.append(latents)
        gini = Oracle(
            expert_trajectories=expert_trajs[:15],
            true_options=true_options[:15],
            expert_trajectories_test=expert_trajs[15:],
            true_options_test=true_options[15:]
        )

    if use_wandb:
        run = wandb.init(
            project=f'{exp_identifier}franka-kitchen',
            name='HBC_{}{}'.format(
                student_type,
                timestamp(),
            ),
            tags=['hbc'],
            config=_config,
            save_code=True,
            monitor_gym=True
        )
    else:
        run = None
    # Create student:
    if student_type=='random':
        student = Random(gini, option_dim=n_targets, query_percent=query_percent)

    elif student_type=='query_cap':
        student = QueryCapLimit(gini, option_dim=n_targets, query_demo_cap=query_cap)

    elif student_type=='iterative_random':
        student = IterativeRandom(gini, option_dim=n_targets)

    elif student_type=='action_entropy':
        student = ActionEntropyBased(gini, option_dim=n_targets)

    elif student_type=='intent_entropy':
        student = IntentEntropyBased(gini, option_dim=n_targets)

    elif student_type=='supervised':
        student = Supervised(gini, option_dim=n_targets)

    elif student_type=='unsupervised':
        student = Unsupervised(gini, option_dim=n_targets)
    
    elif student_type=='action_intent_entropy':
        student = ActionIntentEntropyBased(gini, option_dim=n_targets)
        
    env = gym.make(env_name, 
                   tasks_to_complete=[
                       'microwave',
                    #    'kettle',
                    #    'light switch', 
                    #    'slide cabinet'
                       ])
    # env = Monitor(env)
    env = ObservationOnly(env)
    # Create algorithm with student (who has oracle), env
    hbc = HBC(
        option_dim=n_targets, # check how are the categories selected?
        device='cpu',
        env=env,
        exp_identifier=str(query_percent) + 'query_ratio',
        curious_student=student,
        results_dir='results_fixed_order_targets',
        wandb_run=run
        )

    if student_type in ['action_entropy', 'action_intent_entropy']:
        student.set_policy(hbc)

    if n_epochs is None:
        delay_after_query_exhaustion = 50
        n_epochs = gini.max_queries + delay_after_query_exhaustion
    hbc.train(n_epochs)

    # Log table with metrics
    if run:
        run.log({"metrics/metrics": hbc._logger.metrics_table})
        table = hbc._logger.metrics_table.get_dataframe()
        base_path = os.path.join(os.getcwd(), f'csv_metrics/')
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        file_name = f'{base_path}hbc_{student_type}{timestamp()}.csv'
        table.to_csv(file_name, header=True, index=False)
        wandb.save(file_name, base_path=os.getcwd())

        file_name = file_name.split('.')[0] + '_list_queries.npy'
        hbc.curious_student.save_queries(file_name)
        wandb.save(file_name, base_path=os.getcwd())