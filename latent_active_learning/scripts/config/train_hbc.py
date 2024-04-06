import sacred


train_hbc_ex = sacred.Experiment("train_hbc", interactive=True)


@train_hbc_ex.named_config
def rw4t_discrete():
    env_name = "BoxWorld-v0"
    n_epochs = 100
    use_wandb=True
    num_demos=100
    targets = [
        [ 1, 1 ],
        [ 1, 7 ],
        [ 6, 4 ],
        [ 0, 5 ],
        [ 3, 7 ],
        [ 8, 6 ]
    ]
    danger = [
        [ 0, 2 ],
        [ 1, 1 ],
        [ 2, 8 ],
        [ 4, 7 ],
        [ 7, 3 ],
        [ 7, 4 ],
        [ 7, 6 ]
    ]
    obstacles = [
        [ 4, 0 ],
        [ 5, 0 ],
        [ 2, 1 ],
        [ 5, 1 ],
        [ 8, 1 ],
        [ 3, 2 ],
        [ 8, 2 ],
        [ 1, 3 ],
        [ 5, 3 ],
        [ 6, 3 ],
        [ 1, 4 ],
        [ 2, 4 ],
        [ 5, 4 ],
        [ 8, 4 ],
        [ 5, 5 ],
        [ 8, 5 ],
        [ 1, 6 ],
        [ 2, 6 ],
        [ 3, 6 ],
        [ 0, 7 ],
        [ 2, 7 ],
        [ 6, 7 ],
        [ 8, 7 ],
        [ 8, 8 ],
        [ 4, 9 ],
        [ 5, 9 ]
    ]
    kwargs = {
        'size': 10,
        'n_targets': len(targets),
        'allow_variable_horizon': True,
        'fixed_targets': targets,
        'danger': danger,
        'obstacles': obstacles,
        'latent_distribution': lambda x: 0,
        'render_mode': None
        }


@train_hbc_ex.named_config
def discrete_env():
    env_name = "BoxWorld-v0"
    n_epochs = 100
    use_wandb=True
    num_demos=100

@train_hbc_ex.named_config
def continuous_env():
    env_name = "BoxWorld-continuous-v0"
    n_epochs = 100
    use_wandb=True
    num_demos=100

@train_hbc_ex.named_config
def efficient_learner():
    efficient_student = True

@train_hbc_ex.named_config
def boxworld_2targets():
    n_targets = 2
    kwargs = {
        'size': 5,
        'n_targets': n_targets,
        'allow_variable_horizon': True,
        'fixed_targets': [ [0,0], [4,4] ],
        'latent_distribution': lambda x: 0,
        'render_mode': None
        }

@train_hbc_ex.named_config
def boxworld_3targets():
    n_targets = 3
    kwargs = {
        'size': 5,
        'n_targets': n_targets,
        'allow_variable_horizon': True,
        'fixed_targets': [[0,0],[4,4], [0,4]],
        'latent_distribution': lambda x: 0,
        'render_mode': None
        }

@train_hbc_ex.named_config
def boxworld_4targets():
    n_targets = 4
    n_epochs = 50
    kwargs = {
        'size': 10,
        'n_targets': n_targets,
        'allow_variable_horizon': True,
        'fixed_targets': [[0,0], [9,0] ,[0,9],[9,9]],
        'latent_distribution': lambda x: 0,
        'render_mode': None
        }

@train_hbc_ex.named_config
def boxworld_8targets():
    n_targets = 8
    kwargs = {
        'size': 10,
        'n_targets': n_targets,
        'allow_variable_horizon': True,
        'fixed_targets': [
            [0, 0], [3, 3], [0, 9], [9, 9],
            [6, 3], [9, 0], [3, 6], [6, 6]
            ],
        'latent_distribution': lambda x: 0,
        'render_mode': None
        }

@train_hbc_ex.named_config
def rich_repr():
    filter_state_until=-1

@train_hbc_ex.named_config
def simple_repr(n_targets):
    filter_state_until = -1 - n_targets
