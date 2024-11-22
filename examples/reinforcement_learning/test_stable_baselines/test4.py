import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

import numpy as np
import math
from stable_baselines3.common.env_checker import check_env as sb3_check_env


vec_env = make_vec_env('Acrobot-v1', n_envs=12)

model = PPO("MlpPolicy", vec_env, verbose=1)
model.learn(total_timesteps=250000)
model.save("ppo_acrobot")

del model # remove to demonstrate saving and loading

model = PPO.load("ppo_acrobot")

vec_env = make_vec_env('Acrobot-v1', n_envs=4)

obs = vec_env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")