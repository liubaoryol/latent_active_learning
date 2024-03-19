import sacred


train_hbc_ex = sacred.Experiment("train_hbc", interactive=True)

@train_hbc_ex.named_config
def discrete_env():
    env_name = "BoxWorld-v0"
    n_epochs = 30
    use_wandb=True

@train_hbc_ex.named_config
def continuous_env():
    env_name = "BoxWorld-continuous-v0"
    n_epochs = 30
    use_wandb=True
    
@train_hbc_ex.named_config
def boxworld_2targets():
    n_targets = 2
    kwargs = {
        'size': 5,
        'n_targets': n_targets,
        'allow_variable_horizon': True,
        'fixed_targets': [ [0,0], [4,4] ],
        'latent_distribution': lambda x: 0,
        # 'render_mode': "human"
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

@train_hbc_ex.named_config
def rich_repr():
    filter_state_until=-1

@train_hbc_ex.named_config
def simple_repr(n_targets):
    filter_state_until = -1 - n_targets