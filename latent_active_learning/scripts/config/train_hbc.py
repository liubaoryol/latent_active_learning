import sacred


train_hbc_ex = sacred.Experiment("train_hbc", interactive=True)

@train_hbc_ex.named_config
def default():
    env_name = "BoxWorld-v0"
    filter_state_until = -1
    query_percent = 1
    n_epochs = 15
    
@train_hbc_ex.named_config
def boxworld_2targets():
    n_targets = 2
    kwargs = {
        'size': 5,
        'n_targets': n_targets,
        'allow_variable_horizon': True,
        'fixed_targets': [ [0,0], [4,4] ],
        'latent_distribution': lambda x: 0
        }

@train_hbc_ex.named_config
def boxworld_3targets():
    n_targets = 3
    kwargs = {
        'size': 5,
        'n_targets': n_targets,
        'allow_variable_horizon': True,
        'fixed_targets': [[0,0],[4,4], [0,4]],
        'latent_distribution': lambda x: 0
        }

@train_hbc_ex.named_config
def boxworld_4targets():
    n_targets = 4
    kwargs = {
        'size': 10,
        'n_targets': n_targets,
        'allow_variable_horizon': True,
        'fixed_targets': [[0,0], [9,0] ,[0,9],[9,9]],
        'latent_distribution': lambda x: 0
        }
