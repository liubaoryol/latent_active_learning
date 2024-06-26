"""Wrapper for transforming observations."""
from typing import List
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class FilterLatent(gym.ObservationWrapper, gym.utils.RecordConstructorArgs):
    """Transform the observation via an arbitrary function :attr:`f`.

    The function :attr:`f` should be defined on the observation space of the base environment, ``env``, and should, ideally, return values in the same space.

    If the transformation you wish to apply to observations returns values in a *different* space, you should subclass :class:`ObservationWrapper`, implement the transformation, and set the new observation space accordingly. If you were to use this wrapper instead, the observation space would be set incorrectly.

    Example:
        >>> import gymnasium as gym
        >>> from gymnasium.wrappers import TransformObservation
        >>> import numpy as np
        >>> np.random.seed(0)
        >>> env = gym.make("CartPole-v1")
        >>> env = TransformObservation(env, lambda obs: obs + 0.1 * np.random.randn(*obs.shape))
        >>> env.reset(seed=42)
        (array([0.20380084, 0.03390356, 0.13373359, 0.24382612]), {})
    """

    def __init__(self, env: gym.Env, unobservable_states: List):
        """Initialize the :class:`TransformObservation` wrapper with an environment and a transform function :attr:`f`.

        Args:
            env: The environment to apply the wrapper
            f: A function that transforms the observation
        """
        gym.utils.RecordConstructorArgs.__init__(self, unobservable_states=unobservable_states)
        gym.ObservationWrapper.__init__(self, env)


        wrapped_observation_space = env.observation_space
        if not isinstance(wrapped_observation_space, spaces.Box):
            raise ValueError(
                f"FilterLatent is only usable with Box observations, "
                f"environment observation space is {type(wrapped_observation_space)}"
            )

        self._unobservable_states = unobservable_states
        highs = wrapped_observation_space.high
        lows = wrapped_observation_space.low
        

        self.mask = np.full(len(highs), True, dtype=bool)
        self.mask[self._unobservable_states] = False


        self.observation_space = type(wrapped_observation_space)(
            lows[self.mask], highs[self.mask]
        )

        self._env = env


    def observation(self, observation):
        """Transforms the observations with callable :attr:`f`.

        Args:
            observation: The observation to transform

        Returns:
            The transformed observation
        """

        return observation[self.mask]



class TransformBoxWorldReward(gym.RewardWrapper, gym.utils.RecordConstructorArgs):
    """Transform the reward via an arbitrary function.

    Warning:
        If the base environment specifies a reward range which is not invariant under :attr:`f`, the :attr:`reward_range` of the wrapped environment will be incorrect.

    Example:
        >>> import gymnasium as gym
        >>> from gymnasium.wrappers import TransformReward
        >>> env = gym.make("CartPole-v1")
        >>> env = TransformReward(env, lambda r: 0.01*r)
        >>> _ = env.reset()
        >>> observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
        >>> reward
        0.01
    """

    def __init__(self, env: gym.Env):
        """Initialize the :class:`TransformReward` wrapper with an environment and reward transform function :attr:`f`.

        Args:
            env: The environment to apply the wrapper
            f: A function that transforms the reward
        """
        gym.RewardWrapper.__init__(self, env)

    # def reset(self, seed=None, options=None):
    #     self.visited_goals = []
    #     return self.env.reset()
    def step(self, action):
        """Modifies the :attr:`env` :meth:`step` reward using :meth:`self.reward`."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        observation, reward = self.reward(observation, reward)
        return observation, reward, terminated, truncated, info

    def reward(self, observation, reward):
        """Transforms the reward using callable :attr:`f`.

        Args:
            reward: The reward to transform

        Returns:
            The transformed reward
        """
        
        agent_location = self.unwrapped.occupied_grids[0]
        targets = self.env.unwrapped.occupied_grids[1:]
        for idx, target in enumerate(targets):
            if idx not in self.unwrapped._visited_goals and self.unwrapped.target_achieved(
                agent_location,
                target
            ):
                self.unwrapped._visited_goals.append(idx)
                pos = (self.unwrapped.n_targets+1)* 2 + idx
                if pos < len(observation):
                    observation[pos] = 0
                return observation, 50
        return observation, reward