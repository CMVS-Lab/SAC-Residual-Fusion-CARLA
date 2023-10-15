import carla 
town_name = 'Town01'
# client = carla.Client("localhost", 2000)
# world = client.load_world(town_name)
import gymnasium as gym
from carla_env import CarEnv
import numpy as np
import time

class CARLA_G(gym.Env):
    def __init__(self, ):
        super(CARLA_G, self).__init__()
        self.env = CarEnv()
        self.action_space = gym.spaces.Box(low=-1, high=1, shape = (2, ), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low = -np.inf, high = np.inf, shape=(16, ), dtype=np.float32)

    def step(self, action):
        [new_image, new_state], reward, done, info = self.env.step(action)
        return new_state.astype(np.float32), reward, done, False, {}
    
    def reset(self, seed = None, options = {}):
        image, state = self.env.reset()
        time.sleep(3)
        return state.astype(np.float32), {}
    
    def render(self):
        pass

env = CARLA_G()
# check_env(env, warn=True)
# model = DDPG("MlpPolicy", env)
# model.learn(total_timesteps=10000, log_interval=10)

env.env.town_name = town_name


from stable_baselines3 import SAC, PPO, DDPG
from stable_baselines3.common.env_checker import check_env

model2 = SAC("MlpPolicy", env, verbose=1)

# model = SAC("MlpPolicy", env, verbose=1)
model = SAC.load("SAC_model_run3_97", print_system_info=True)
model.env = model2.env
# 81
# for i in range(97,1000):
#     name = "SAC_model_run3_"+str(i+1)
#     print(name)
#     model.learn(total_timesteps=10000, log_interval=4)
#     model.save(name)
#     # time.sleep(60)

#env.env.world = env.env.client.load_world(env.env.town_name)
obs, info = env.reset()
# Evaluate the agent
episode_reward = 0
for _ in range(20000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    episode_reward += reward
    if terminated or truncated or info.get("is_success", False):
        print("Reward:", episode_reward, "Success?", info.get("is_success", False))
        episode_reward = 0.0
        obs, info = env.reset()