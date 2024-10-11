# Import runs
import pandas as pd
import numpy as np
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
# Important line to avoid core abort. Do not remove
figure, axis = plt.subplots(2, 1)

from latent_active_learning.oracle import Oracle

entity = 'vinson-liuba-experiments'
project = 'BoxWorld-size10-targets6'
# entity = 'liuba1156'
# project = 'rw4t-discrete-looks-promising'
num_epochs_trained_on_all = {
    'BoxWorld-size10-targets6': 3500,
    'BoxWorld-size10-targets3': 2000,
    'BoxWorld-size10-targets2': 650,
    'rw4t-discrete-new-good3': 10,
    'rw4t-discrete': 5363,
    'rw4t-discrete-looks-promising': 500
}
api = wandb.Api()
runs = api.runs(entity + "/" + project)

counter = 0
dataframes = []
list_queries = []
for run in runs:
    for file in run.files():
        if '.csv' in file.name:
            counter +=1
            print("Downloading", counter)
            file.download(replace=True)
            df = pd.read_csv(file.name)
            df['seed'] = run.config['seed']
            dataframes.append(df)
        if '.npy' in file.name:
            counter +=1
            print("Downloading", counter)
            file.download(replace=True)
            list_queries.append(np.load(file.name, allow_pickle=True))
            

for idx, query in enumerate(list_queries):
    print(idx, len(query) == len(set(query)))
    assert len(query) == len(set(query))

rows = []
for df in dataframes:
    rows.append(len(df))

for idx, df in enumerate(dataframes):
    dataframes[idx] = df[:num_epochs_trained_on_all[project]]

from copy import deepcopy as copy
# path ="/home/liubove/Documents/my-packages/rw4t-dataset/" \
#     "dataset/trajectories/discrete/gini_n18"
# gini = Oracle.load(path)
# # gini = Oracle.load('./expert_trajs/{}'.format(project))
# roll_train, roll_test = gini.stats()

# expert_df = copy(dataframes[0])
# expert_df['student_type']='expert'
# expert_df['rollout_mean'] = 1# roll_train[0]
# expert_df['rollout_std'] = 0 # roll_train[1]
# expert_df['hamming_train'] = 0
# expert_df['hamming_test'] = 0
# expert_df['prob_true_action'] = 0
# expert_df['prob_true_action_test'] = 0
# dataframes.append(expert_df)

full_df = pd.concat(dataframes, ignore_index=True)
full_df = full_df.rename(columns={'epoch': 'iteration', 'student_type': 'algorithm'})

def set_lineplot_ax(df, ax, metric='performance', legend=False, loc_legend='lower right', kwargs=None):
    # For location parameters see here:
    # https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.legend.html#matplotlib.axes.Axes.legend
    # Set the dashing
    # bool, list, dict: (segment, gap) or '' for solid line

    sns.lineplot(data=df,
                 ax=ax,
                 x='iteration',
                 y=metric,
                 hue='algorithm',
                 style='algorithm',
                 **kwargs
                 )
    # ax.set_xlim(0,1001)
    # ax.set_ylim(3,8)
    if legend:
        ax.legend(title=None, loc=loc_legend)
    else:
        ax.legend_.remove()
    ax.set_xlabel( "Iteration", size=12 )
    ax.set_ylabel( metric, size=12)


dashes = {alg: '' for alg in set(full_df['algorithm'])}
markers = {alg: ',' for alg in set(full_df['algorithm'])}
dashes['expert'] = (3, 2)
markers['expert'] = ','
LINEPLOT_FLAVOR_BANDS = {
    'dashes': dashes,
    'markers': markers,
    'err_kws': {'alpha': 0.15} # set saturation of bands
}

LINEPLOT_FLAVOR_BARS = {
    'dashes': dashes,
    'markers': markers,
    'err_style': 'bars',
    'err_kws': {'capsize': 3},
    'palette': 'colorblind'
}


# # Normalize
# metric = full_df['rollout_mean']
# full_df['rollout_mean'] = (metric - min(metric)) / (1961.7767333984375 - min(metric)) 
# metric='rollout_mean'

def select_epochs_evently(full_df, num_epochs, num_expts=25):
    total_iterations = full_df['iteration'].iloc[-1]
    epoch_selection = np.linspace(0, total_iterations, num=num_epochs, dtype=int)
    indexes = np.zeros(total_iterations+1, bool)
    indexes[epoch_selection] = 1
    indexes = np.concatenate([indexes]*num_expts)
    return full_df[indexes]


df = select_epochs_evently(full_df, 20, 21)


figure, axis = plt.subplots(1, 1)

set_lineplot_ax(df, axis, metric=metric, legend=True, kwargs=LINEPLOT_FLAVOR_BANDS)

# sns.set_style('whitegrid')
sns.set_style('white')
sns.color_palette("Paired")
plt.show()
