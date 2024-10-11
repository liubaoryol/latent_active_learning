from stable_baselines3.common.utils import obs_as_tensor
import numpy as np
import torch
from imitation.algorithms import base as algo_base

hbc.train(10)
self = hbc
transitions_lo, transitions_hi = self.transitions(self.curious_student.demos) # or demos_test
# transitions_lo, transitions_hi = self.transitions(self.curious_student.demos_test)
# log_prob have to be higher, check what are the actions exactly/

# Make a function to get stats at each state
obs = transitions_lo.obs
acts = transitions_lo.acts
predicted_acts, values, log_prob =  hbc.policy_lo.policy(obs_as_tensor(transitions_lo.obs, 'cpu'))
acts2, _ = hbc.policy_lo.policy.predict(obs_as_tensor(obs, 'cpu'))
def get_stats(at_state, obs=obs, acts=acts):
    if isinstance(acts, torch.Tensor):
        acts = acts.numpy()
    idxs = np.where((obs == at_state).all(1))[0]
    acts_used = {}
    for idx in idxs:
        act = acts[idx]
        if act not in acts_used:
            acts_used[act] = 0
        acts_used[act] += 1
    return acts_used

def get_comparison(at_state, acts_true, acts_predicted):
    num_acts_used = get_stats(at_state, acts=acts_true)
    num_acts_predicted = get_stats(at_state, acts=acts_predicted)
    return num_acts_used, num_acts_predicted

stats = []
for at_state in transitions_lo.obs:
    stats_true, stats_compare = get_comparison(at_state, acts_true=acts, acts_predicted=predicted_acts)
    
    stats.append((stats_true, stats_compare))


hbc.policy_lo._demo_data_loader = algo_base.make_data_loader(
            transitions_lo,
            32,
            data_loader_kwargs={'shuffle':False})

rs  = next(iter(hbc.policy_lo._demo_data_loader))

observation = env.reset()[0]
actions, state = hbc.predict(observation.reshape(1,-1), state, episode_start=False)
observation = env.step(actions.item())[0]


hbc.policy_lo.policy(torch.Tensor([observation.astype(int).tolist()+[4]]),)