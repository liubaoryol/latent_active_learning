import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import torch

class BoxWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5, n_targets=1,
                 latent_distribution=np.random.randint,
                 allow_variable_horizon=True,
                 fixed_targets=None):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window
        self._max_episode_steps = n_targets**2 * size**3

        self.n_targets = n_targets
        self.fixed_targets=fixed_targets
        self.occupied_grids = np.empty((n_targets + 1, 2), dtype=np.int64)
        self.latent_distribution = latent_distribution

        self.allow_variable_horizon=allow_variable_horizon
        self.at_absorb_state = False
        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        # self.observation_space = spaces.Dict(
        #     {
        #         "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
        #         "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
        #     }
        # )

        obs_shape = 2*(n_targets + 1)
        
        highs = [ size-1 ] * obs_shape
        highs += [ 1 ] * n_targets
        highs.append(n_targets - 1)

        self.observation_space = spaces.Box(0, np.array(highs), shape=(obs_shape + n_targets + 1,), dtype=int)

        # We have 4 actions, corresponding to "right", "up", "left", "down", "right"
        self.action_space = spaces.Discrete(4)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        visited = [0 if t in self._visited_goals else 1 for t in range(self.n_targets)]

        obs = np.concatenate(self.occupied_grids)
        obs = np.concatenate([obs, visited, [self._curr_goal]])

        return obs

    def get_obs(self):
        visited = [0 if t in self._visited_goals else 1 for t in range(self.n_targets)]

        obs = np.concatenate(self.occupied_grids)
        obs = np.concatenate([obs, visited, [self._curr_goal]])

        return obs

    # def _get_info(self):
    #     return {
    #         "distance": np.linalg.norm(
    #             self._agent_location - self._target_location, ord=1
    #         )
    #     }

    def _is_occupied(self, location):
        return (self.occupied_grids==location).all(axis=1).any()

    def reset(self, seed=None, options=None):
        return self._reset(seed=seed, targets=self.fixed_targets)

    def _reset(self, seed=None, targets=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.at_absorb_state = False

        self.occupied_grids = np.empty((self.n_targets + 1, 2), dtype=np.int64)
        self._elapsed_steps = 0
        self._visited_goals = []
        self._curr_goal = self.sample_next_goal()

        if targets is not None:
            assert len(targets)==self.n_targets, \
            'Number of targets should be {}, but {} were given'.format(
                self.n_targets, len(targets))
            self.occupied_grids[1:] = targets
            
        else:
            # We will sample the target's location randomly until it does not coincide with the agent's location
            for element in range(1, self.n_targets+1):
                target_location = self.occupied_grids[0]
                while self._is_occupied(target_location):
                    target_location = self.np_random.integers(
                        0, self.size, size=2, dtype=int
                    )
                self.occupied_grids[element] = target_location

        # Choose the agent's location uniformly at random
        agent_location = self.np_random.integers(0, self.size, size=2)
        while self._is_occupied(agent_location):
            agent_location = self.np_random.integers(0, self.size, size=2)

        self.occupied_grids[0] = agent_location


        observation = self._get_obs()
        info = {}
        # info = self._get_info()

        if self.render_mode=="human":
            self._render_frame()
        elif self.render_mode=="rgb_array":
            observation = self._render_frame()

        return observation, info

    def step(self, action):
        truncated = False
        terminated = False

        self._elapsed_steps += 1

        if self.at_absorb_state:
            reward = 0

        else:
            if len(self._visited_goals) == self.n_targets:
                raise ValueError("All targets have been visited. Call `reset()` function")

            reward = -1
            # Map the action (element of {0,1,2,3}) to the direction we walk in
            direction = self._action_to_direction[action]
            # We use `np.clip` to make sure we don't leave the grid
            agent_location = self.occupied_grids[0]
            agent_location = np.clip(
                agent_location + direction, 0, self.size - 1
            )
            self.occupied_grids[0] = agent_location

            curr_target = self.occupied_grids[self._curr_goal + 1]
            # An episode is done iff the agent has reached the target
            if self.target_achieved(agent_location, curr_target):
                self._visited_goals.append(self._curr_goal)
                reward = 50
                if len(self._visited_goals) == self.n_targets:
                    if self.allow_variable_horizon:
                        terminated = True
                    else:
                        self.at_absorb_state = True
                else:
                    self._curr_goal = self.sample_next_goal()
        
        if self._elapsed_steps >= self._max_episode_steps:
            truncated = True
            terminated = True

        observation = self._get_obs()
        info = {}
        # info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()
        elif self.render_mode=="rgb_array":
            observation = self._render_frame()

        return observation, reward, terminated, truncated, info

    def target_achieved(self, agent_location, target):
        return np.array_equal(agent_location, target)

    def step_from_obs(self, obs, action):

        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        agent_location = obs[:2]
        agent_location = np.clip(
            agent_location + direction, 0, self.size - 1
        )
        return np.concatenate([agent_location, obs[2:]])

    def sample_next_goal(self):
        targets = list(range(self.n_targets))
        for visited in self._visited_goals:
            targets.remove(visited)
        m = len(targets)
        # if m > 0
        return targets[self.latent_distribution(m)]

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        agent_location = self.occupied_grids[0]
        targets = self.occupied_grids[1:]
        for idx, target_location in enumerate(targets):
            if idx not in self._visited_goals:
                color = (155, 0, 0) if idx==self._curr_goal else (255, 0, 0)
                pygame.draw.rect(
                    canvas,
                    color,
                    pygame.Rect(
                        pix_square_size * target_location,
                        (pix_square_size, pix_square_size),
                    ),
                )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(2, 1, 0)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None



# ############## COLLECT DEMO
# # import gymnasium as gym
# # import latent_active_learning

# # from imitation.data import rollout
# # from imitation.scripts.ingredients import policy_evaluation
# # import imitation.data.serialize as data_serialize
# # from stable_baselines3.common.env_util import make_vec_env
# # from stable_baselines3 import PPO

# # rl_algo = PPO.load("boxworld2targets")
# # rl_algo.env = make_vec_env("envs/BoxWorld-v0", n_envs=4, env_kwargs={'n_targets': 2})

# # rollout_save_n_timesteps = 4
# # rollout_save_n_episodes = 100
# # sample_until = rollout.make_sample_until(
# #     rollout_save_n_timesteps,
# #     rollout_save_n_episodes,
# # )
# # rng = np.random.default_rng()
# # trajs = rollout.rollout(rl_algo, rl_algo.get_env(), sample_until, rng=rng, unwrap=False)
# # save_path = "trajectories_final.npz"
# # data_serialize.save(save_path, trajs)
# # policy_evaluation.eval_policy(rl_algo, venv, 10, rng)
# # rollouts = rollout.flatten_trajectories(trajs)



def trajs2demo(trajswR):
    sample = []
    for traj in trajswR:
        sample.append((
            torch.Tensor(traj.obs[:-1]),
            torch.Tensor(traj.acts),
            torch.Tensor(traj.rews)
        ))
    return sample