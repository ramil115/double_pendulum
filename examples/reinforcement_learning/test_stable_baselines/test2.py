import gymnasium as gym


class BasicWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.observation, _ = self.reset()
        
    def step(self, action):        
        next_state, reward, terminated, truncated, info = self.env.step(action)
        self.observation = next_state
        reward = self.reward_func(self.observation, action)

        return next_state, reward, terminated, truncated, info

    def reward_func(self, observation, action):
        return observation[0]


# env = gym.make('MountainCarContinuous-v0', render_mode="human")
# env_w = BasicWrapper(env)

# Parallel environments
envs = gym.make_vec('MountainCarContinuous-v0', num_envs=4)
# vec_env_w = BasicWrapper(vec_env)
