"""Wrapper for transforming observations."""
from typing import List
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# from aic_domain.box_push.maps import EXP1_MAP
# from aic_domain.box_push.mdp import BoxPushTeamMDP_AlwaysTogether


class MoversAdapt(gym.ObservationWrapper, gym.utils.RecordConstructorArgs):
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

    def __init__(self, env: gym.Env):
        """Initialize the :class:`TransformObservation` wrapper with an environment and a transform function :attr:`f`.

        Args:
            env: The environment to apply the wrapper
            f: A function that transforms the observation
        """
        gym.ObservationWrapper.__init__(self, env)

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,))

        self._env = env


    def observation(self, observation):
        """Transforms the observations with callable :attr:`f`.

        Args:
            observation: The observation to transform

        Returns:
            The transformed observation
        """
        tup_state = self.unwrapped.mdp.conv_mdp_sidx_to_sim_states(observation)
        return np.concatenate(tup_state)



class MoversFullyObs(gym.ObservationWrapper, gym.utils.RecordConstructorArgs):
    """Transform the observation via an arbitrary function :attr:`f`. Add robot's state to obs

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

    def __init__(self, env: gym.Env):
        """Initialize the :class:`TransformObservation` wrapper with an environment and a transform function :attr:`f`.

        Args:
            env: The environment to apply the wrapper
            f: A function that transforms the observation
        """
        gym.ObservationWrapper.__init__(self, env)

        # self.mdp = BoxPushTeamMDP_AlwaysTogether(**EXP1_MAP)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,))

        self._env = env

        self.OPTION_DICT = {
            ('pickup', 0): 0,
            ('pickup', 1): 1,
            ('pickup', 2): 2,
            ('goal', 0): 3
        }


    def observation(self, observation):
        """Transforms the observations with callable :attr:`f`.

        Args:
            observation: The observation to transform

        Returns:
            The transformed observation
        """
        tup_state = self.unwrapped.mdp.conv_mdp_sidx_to_sim_states(observation)
        r_goal = self._env.unwrapped.robot_agent.get_current_latent()
        r_goal = self.OPTION_DICT[r_goal]

        state = np.concatenate(tup_state)
        return np.concatenate([state, [r_goal]])

class MoversBoxWorldRepr(gym.ObservationWrapper, gym.utils.RecordConstructorArgs):
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

    def __init__(self, env: gym.Env):
        """Initialize the :class:`TransformObservation` wrapper with an environment and a transform function :attr:`f`.

        Args:
            env: The environment to apply the wrapper
            f: A function that transforms the observation
        """
        gym.ObservationWrapper.__init__(self, env)

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,))

        self._env = env


    def observation(self, observation):
        """Transforms the observations with callable :attr:`f`.

        Args:
            observation: The observation to transform

        Returns:
            The transformed observation
        """
        tup_state = self.unwrapped.mdp.conv_mdp_sidx_to_sim_states(observation)
        return np.concatenate(tup_state[1:])


