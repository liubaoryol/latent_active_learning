import sacred


train_hbc_ex = sacred.Experiment("train_hbc", interactive=True)


@train_hbc_ex.named_config
def rw4t_discrete():
    env_name = "BoxWorld-v0"
    n_targets = 6
    n_epochs = 3500
    use_wandb=True
    filter_state_until = -1 - n_targets
    kwargs = {
        'size': 10,
        'n_targets': n_targets,
        'allow_variable_horizon': True,
        'fixed_targets': [
            [ 1, 1 ],
            [ 7, 1 ],
            [ 4, 6 ],
            [ 5, 0 ],
            [ 7, 3 ],
            [ 6, 8 ],
        ],
        'danger': [
            [ 2, 0 ],
            [ 1, 1 ],
            [ 8, 2 ],
            [ 7, 4 ],
            [ 3, 7 ],
            [ 4, 7 ],
            [ 6, 7 ],
        ],
        'obstacles': [
            [ 0, 4 ],
            [ 0, 5 ],
            [ 1, 2 ],
            [ 1, 5 ],
            [ 1, 8 ],
            [ 2, 3 ],
            [ 2, 8 ],
            [ 3, 1 ],
            [ 3, 5 ],
            [ 3, 6 ],
            [ 4, 1 ],
            [ 4, 2 ],
            [ 4, 5 ],
            [ 4, 8 ],
            [ 5, 5 ],
            [ 5, 8 ],
            [ 6, 1 ],
            [ 6, 2 ],
            [ 6, 3 ],
            [ 7, 0 ],
            [ 7, 2 ],
            [ 7, 6 ],
            [ 7, 8 ],
            [ 8, 8 ],
            [ 9, 4 ],
            [ 9, 5 ],
        ],
        'latent_distribution': lambda x: 0,
        'render_mode': None
        }


@train_hbc_ex.named_config
def discrete_2targets_boxworld():
    env_name = "BoxWorld-v0"
    n_targets = 2
    n_epochs = 650
    use_wandb=True
    filter_state_until = -1 - n_targets

    kwargs = {
        'size': 10,
        'n_targets': n_targets,
        'allow_variable_horizon': True,
        'fixed_targets': [ [0, 0], [9, 9] ],
        'latent_distribution': lambda x: 0,
        'render_mode': None
        }


@train_hbc_ex.named_config
def movers():
    env_name = "EnvMovers-v0"
    # fixed_latent=True
    movers_optimal=False
    options_w_robot=False
    state_w_robot_opts = False
    fixed_latent=False
    if options_w_robot:
        n_targets = 16
    else:    
        n_targets = 4
    use_wandb = True


@train_hbc_ex.named_config
def discrete_3targets_boxworld():
    env_name = "BoxWorld-v0"
    n_targets = 3
    n_epochs = 2000
    use_wandb=True
    filter_state_until = -1 - n_targets

    kwargs = {
        'size': 10,
        'n_targets': n_targets,
        'allow_variable_horizon': True,
        'fixed_targets': [ [0, 0], [9, 9], [4, 5] ],
        'latent_distribution': lambda x: 0,
        'render_mode': None
        }

@train_hbc_ex.named_config
def continuous_env():
    env_name = "BoxWorld-continuous-v0"
    use_wandb=True
    num_demos=100

@train_hbc_ex.named_config
def efficient_learner():
    efficient_student = True

@train_hbc_ex.named_config
def boxworld_4targets():
    n_targets = 4
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
