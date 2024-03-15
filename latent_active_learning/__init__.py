from gymnasium.envs.registration import register

register(
    id="BoxWorld-v0",
    entry_point="latent_active_learning.envs:BoxWorldEnv",
)

register(
    id="BoxWorld-continuous-v0",
    entry_point="latent_active_learning.envs:BoxWorldContinuousEnv",
)