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

env = gym.make('MountainCarContinuous-v0', render_mode="human")
env = BasicWrapper(env)
# observation, info = env.reset()

episode_over = False
while not episode_over:
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)
    print(reward - observation[0])

    episode_over = terminated or truncated

env.close()