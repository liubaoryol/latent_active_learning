import numpy as np
import gymnasium as gym

# import imitation
# from aic_domain.box_push.maps import EXP1_MAP
# from aic_domain.box_push.mdp import BoxPushTeamMDP_AlwaysTogether
# from aic_domain.box_push.policy import BoxPushPolicyTeamExp1
# from aic_domain.box_push.agent import BoxPushAIAgent_Team1

from latent_active_learning.scripts.config.train_hbc import train_hbc_ex
from latent_active_learning.collect import get_trajectories
from latent_active_learning.collect import filter_TrajsWRewards
from latent_active_learning.collect import get_expert
# from latent_active_learning.wrappers.movers_wrapper import MoversAdapt, MoversBoxWorldRepr
# from latent_active_learning.wrappers.movers_wrapper import MoversFullyObs


@train_hbc_ex.capture
def get_demos(env_name, filter_state_until, kwargs, num_demos):
    '''Get clean rollouts and corresponding of simple BoxWorld'''
    expert = get_expert(env_name, kwargs)

    rollouts_full = get_trajectories(
        env_name=env_name,
        model=expert,
        kwargs=kwargs,
        n_demo=num_demos
        )

    rollouts = filter_TrajsWRewards(rollouts_full, filter_state_until)
    options = [roll.obs[:,-1] for roll in rollouts_full]

    return rollouts, options


def concat_obslat(rollouts, options):
    '''Append option to observation'''
    trajs = []
    for idx, demo in enumerate(rollouts):
        tmp = imitation.data.types.TrajectoryWithRew(
            obs = np.concatenate(
                [demo.obs, np.expand_dims(options[idx], 1)], 1),
            acts = demo.acts,
            infos = demo.infos,
            terminal = demo.terminal,
            rews = demo.rews
        )
        trajs.append(tmp)
    return trajs


def get_movers_demos(num_demos,
                     optimal_trj: bool,
                     opts_incl_robot: bool,
                     state_w_robot_opts:bool,
                     fixed_latent: bool,
                     box_repr: bool):
    rollouts = []
    options = []
    
    threshold = -65 if optimal_trj else -200
    
    for idx in range(num_demos):
        print('Collecting demo: ', idx, '...')
        acc_rew = -250
        while acc_rew < threshold:
            roll, opts, acc_rew = get_movers_one_demo(opts_incl_robot,
                                                      state_w_robot_opts,
                                                      fixed_latent,
                                                      box_repr)
        rollouts.append(roll)
        options.append(opts)

    return rollouts, options


OPTION_SEQ = [0,3,2,3,1,3]
def get_movers_one_demo(opts_incl_robot: bool, state_w_robot_opts: bool, fixed_latent: bool, box_repr: bool):
    option_idx = 0
    assert not (opts_incl_robot and state_w_robot_opts), 'Robot options\
        can only be in either state, or options, but not both'
    
    if state_w_robot_opts:
        env = MoversFullyObs(gym.make('EnvMovers-v0'))
    else:
        env = MoversAdapt(gym.make('EnvMovers-v0'))
    
    expert = MoversExpert()
    other_agent= env.unwrapped.robot_agent
    state, infos = env.reset() # type: np.ndarray
    expert.init_latent(state)
    if fixed_latent:
        while other_agent.get_current_latent()[1] != OPTION_SEQ[0]:
            other_agent.init_latent(([0,0,0],(6,2),(6,4)))
        while expert.agent.get_current_latent()[1] != OPTION_SEQ[0]:
            expert.init_latent(state)
    expert.log_latent()
    option_idx = 2

    traj = RolloutStorage(first_state = state)
    
    
    expert.add_robot_option(other_agent.get_current_latent())
    done = False
    acc_rew = 0
    while not done:
    # for _ in range(150):
        action = expert.get_action(state)

        next_state, rew, done, _, info = env.step(action)
        traj.add_data(next_state, action, rew, done)

        expert.update_mental_state(state, action, next_state)

        if option_idx == 2:
            if next_state[0] == 4 and fixed_latent:
                while expert.agent.get_current_latent()[1] != OPTION_SEQ[option_idx]:
                    expert.update_mental_state(state, action, next_state)
                while other_agent.get_current_latent()[1] != OPTION_SEQ[option_idx]:
                    other_agent.init_latent(([0,0,0],(6,2),(6,4)))
                option_idx = 4
        if option_idx == 4:
            if next_state[2] == 4 and fixed_latent:
                while expert.agent.get_current_latent()[1] != OPTION_SEQ[option_idx]:
                    expert.update_mental_state(state, action, next_state)
                while other_agent.get_current_latent()[1] != OPTION_SEQ[option_idx]:
                    other_agent.init_latent(([0,0,0],(6,2),(6,4)))
                option_idx = 6
        expert.log_latent()
        expert.add_robot_option(other_agent.get_current_latent())

        acc_rew += rew
        if done:
            break
        state = next_state
    return traj.get_traj(box_repr), expert.get_options(opts_incl_robot), acc_rew


class MoversExpert():
    def __init__(self):
        self.mdp = BoxPushTeamMDP_AlwaysTogether(**EXP1_MAP)
        policy = BoxPushPolicyTeamExp1(self.mdp, temperature=0.3, agent_idx=0)
        self.agent = BoxPushAIAgent_Team1(policy)
        self.action_dict = {}
        self._options = []
        self._robot_options = []

        self.OPTION_DICT = {
            ('pickup', 0): 0,
            ('pickup', 1): 1,
            ('pickup', 2): 2,
            ('goal', 0): 3
        }
    
    @property
    def options(self):
        return np.array(self._options)
    
    @property
    def robot_options(self):
        return np.array(self._robot_options)

    @property
    def unique_options(self):
        options = np.stack((self.options, self.robot_options), axis=1)
        options = [tuple(opt) for opt in options]
        return set(options)

    def get_action(self, state) -> int:
        # State will be a np.ndarray
        # Hast to be converted to a tup_state
        tup_state = (list(state[:3]), tuple(state[3:5]), tuple(state[5:7]))
        action = self.agent.get_action(tup_state)
        action_idx = self.mdp.a1_a_space.action_to_idx[action]
        if action_idx not in self.action_dict:
            self.action_dict[action_idx] = action
        return action_idx

    def init_latent(self, state):
        tup_state = (list(state[:3]), tuple(state[3:5]), tuple(state[5:7]))
        self.agent.init_latent(tup_state)

    def log_latent(self):
        option = self.agent.get_current_latent()
        self._options.append(self.OPTION_DICT[option])

    def update_mental_state(self, state, action, next_state):
        tup_state = (list(state[:3]), tuple(state[3:5]), tuple(state[5:7]))
        tup_next = (list(next_state[:3]), tuple(next_state[3:5]), tuple(next_state[5:7]))
        a = self.action_dict[action]
        self.agent.update_mental_state(tup_state, (a,), tup_next)

    def add_robot_option(self, option):
        self._robot_options.append(self.OPTION_DICT[option])

    def get_options(self, opts_incl_robot=True):
        if not opts_incl_robot:
            return self.options
        import itertools
        comb = itertools.product(range(4), range(4))
        flat_rule = {c: idx for idx, c in enumerate(comb)}
        options = np.stack((self.options, self.robot_options), axis=1)
        options = [flat_rule[tuple(opt)] for opt in options]
        return np.array(options)
        

class RolloutStorage():
    def __init__(self, first_state):
        self.obs = [first_state]
        self.acts = []
        self.rews = []

    def add_data(self, o, a, r, d, robot_st=None):
        if robot_st is not None:
            o = np.concatenate([o, robot_st])
        self.obs.append(o)
        self.acts.append(a)
        self.rews.append(r)
        self.done = d

    def get_traj(self, box_repr):
        if box_repr:
            obs = np.array(self.obs)[:, 3:]
        else:
            obs = np.array(self.obs)
        trj = imitation.data.types.TrajectoryWithRew(
            obs = obs,
            acts = np.array(self.acts),
            infos = None,
            terminal = self.done,
            rews = np.array(self.rews).astype(float)
        )
        return trj