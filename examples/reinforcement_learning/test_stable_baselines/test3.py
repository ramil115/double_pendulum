import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from gymnasium.envs.classic_control.continuous_mountain_car import Continuous_MountainCarEnv
import numpy as np
import math
from stable_baselines3.common.env_checker import check_env as sb3_check_env


class MyCar(Continuous_MountainCarEnv):

    def step(self, action: np.ndarray):

        position = self.state[0]
        velocity = self.state[1]
        force = min(max(action[0], self.min_action), self.max_action)

        velocity += force * self.power - 0.0025 * math.cos(3 * position)
        if velocity > self.max_speed:
            velocity = self.max_speed
        if velocity < -self.max_speed:
            velocity = -self.max_speed
        position += velocity
        if position > self.max_position:
            position = self.max_position
        if position < self.min_position:
            position = self.min_position
        if position == self.min_position and velocity < 0:
            velocity = 0

        # Convert a possible numpy bool to a Python bool.
        terminated = bool(
            position >= self.goal_position and velocity >= self.goal_velocity
        )

        reward = 0
        if terminated:
            reward = 100.0
        # reward -= math.pow(action[0], 2) * 0.1
        if position > 0:
            reward += position*10

        self.state = np.array([position, velocity], dtype=np.float32)

        if self.render_mode == "human":
            self.render()
        return self.state, reward, terminated, False, {}

gym.register(
    id="MyCar",
    entry_point=MyCar,
    max_episode_steps=999,
    reward_threshold=90.0,
)


# env = MyCar(render_mode="human")
# env = gym.make('MyCar', render_mode="human")

# sb3_check_env(env)
# print("StableBaselines3 env_check successful.")

# Parallel environments

# vec_env = make_vec_env(MyCar, n_envs=10,env_kwargs = {"render_mode":"rgb_array"})
vec_env = make_vec_env('MyCar', n_envs=12)

model = PPO("MlpPolicy", vec_env, verbose=1)
model.learn(total_timesteps=250000)
model.save("ppo_mountain")

del model # remove to demonstrate saving and loading

model = PPO.load("ppo_mountain")

vec_env = make_vec_env('MyCar', n_envs=4)

obs = vec_env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")