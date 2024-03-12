import gymnasium as gym
import numpy as np

from latent_active_learning.collect import get_trajectories
from latent_active_learning.collect import filter_TrajsWRewards
from latent_active_learning.collect import get_expert
from latent_active_learning.wrappers.latent_wrapper import FilterLatent
from latent_active_learning.hbc import HBC
from latent_active_learning.oracle import Random, Oracle


ENV_NAME = 'BoxWorld-v0'
N_TARGETS = 2
KWARGS = {
    'size': 5,
    'n_targets': N_TARGETS,
    'allow_variable_horizon': True,
    'fixed_targets': [ [0, 0], [4, 4] ]
    }

FILTER_STATE_UNTIL = -1 - N_TARGETS

def simple_setup():
    '''Get clean rollouts and corresponding of simple BoxWorld'''
    expert = get_expert(ENV_NAME, KWARGS)

    rollouts_full = get_trajectories(
        env_name=ENV_NAME,
        model=expert,
        kwargs=KWARGS,
        n_demo=500
        )

    rollouts = filter_TrajsWRewards(rollouts_full, FILTER_STATE_UNTIL)
    options = [roll.obs[:,-1] for roll in rollouts_full]

    return rollouts, options


def test_query_oracle():
    """Tests a 100% querying of student to oracle"""
    rollouts, options = simple_setup()
    gini = Oracle(rollouts, options)
    student = Random(rollouts, gini, option_dim=N_TARGETS, query_percent=1)

    student.query_oracle()

    true_options_1= gini.true_options[0]
    assert (student.demos[0]._latent[1:] == true_options_1).all()
    assert (student.demos[0]._is_latent_estimated == 1).all()


def test_hbc_training_supervised():
    """Test for training of HBC with full observability of options"""
    rollouts, options = simple_setup()
    gini = Oracle(rollouts, options)
    student = Random(rollouts, gini, option_dim=N_TARGETS, query_percent=1)
    
    student.query_oracle()

    env = gym.make("BoxWorld-v0", **KWARGS)
    hbc = HBC(
        option_dim=N_TARGETS,
        device='cpu',
        env=FilterLatent(env, list(range(FILTER_STATE_UNTIL, 0))),
        exp_identifier='0.5ratio_query',
        curious_student=student,
        results_dir='results_fixed_order_random_targets'
        )
    hbc.train(15)

    assert hbc._logger.last_mean > 80
    assert np.isclose(hbc._logger.last_std, 0, atol=2)
    assert hbc._logger.last_01_distance == 0


def test_viterbi():
    """Test viterbi on queried options and on trained hbc policies"""
    rollouts, options = simple_setup()
    gini = Oracle(rollouts, options)
    student = Random(rollouts, gini, option_dim=N_TARGETS, query_percent=1)
    student.query_oracle()
    
    # Test Viterbi with 100% of option labels available
    true_options_1= gini.true_options[0]
    assert (student.demos[0].latent[1:].squeeze() == true_options_1).all()

    env = gym.make("BoxWorld-v0", **KWARGS)
    hbc = HBC(
        option_dim=N_TARGETS,
        device='cpu',
        env=FilterLatent(env, list(range(FILTER_STATE_UNTIL, 0))),
        exp_identifier='0.5ratio_query',
        curious_student=student,
        results_dir='results_fixed_order_random_targets'
        )
    hbc.train(10)

    # Test viterbi on expert high and low policies
    student = Random(rollouts, gini, option_dim=N_TARGETS, query_percent = 0)
    
    true_options_1= gini.true_options[0]
    # TODO: almost all is enough, not all specifically.
    assert (student.demos[0].latent[1:].squeeze() == true_options_1).all()
    


# env = gym.make("BoxWorld-v0", **KWARGS)
# hbc = HBC(
#     option_dim=N_TARGETS,
#     device='cpu',
#     env=FilterLatent(env, list(range(FILTER_STATE_UNTIL, 0))),
#     exp_identifier='0.5ratio_query',
#     curious_student=student,
#     results_dir='results_fixed_order_random_targets'
#     )


#     env_name = "BoxWorld-v0"
    # n_targets = 2
    # target_selection = lambda x: 0
    # kwargs = {
    #     'size': 5,
    #     'n_targets': n_targets,
    #     'allow_variable_horizon': True,
    #     'fixed_targets': [[0,0],[4,4]], # NOTE: for 2 targets
    #     # 'fixed_targets': [[0,0],[4,4], [0,4]] # NOTE: for 3 targets
    #     # 'fixed_targets': [[0,0], [9,0] ,[0,9],[9,9]] # NOTE: FOR 4 targets. Remember to change size to 10
    #     # 'fixed_targets': [[0,0],[4,4], [0,4], [9,0], [0,9],[9,9]], # NOTE: for 6 targets. Remember to change size to 10
    #     # 'fixed_targets': None,
    #     'latent_distribution': target_selection
    #     }