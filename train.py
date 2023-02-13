

from stable_baselines3 import A2C,PPO
from stable_baselines3.common.env_util import make_vec_env
from gym_data.envs.Carcassone_env import CarcassoneEnv

from gym.envs.registration import register
import gym 
import torch


# device = torch.device("cuda")
register(id='Carcassone-v0',entry_point='gym_data.envs:CarcassoneEnv',) 

# Parallel environments
env = make_vec_env('gym_data:Carcassone-v0',n_envs =64)

# from stable_baselines3.common.env_checker import check_env
# check_env(env)

model = A2C("MultiInputPolicy", env, verbose=1,device="auto",learning_rate=0.001)

model.set_parameters("carcassone_model11")
model.learn(total_timesteps=10000000)
model.save("carcassone_model12")
# model.to(device)

obs = env.reset()

for i in range(500):

    action, _state = model.predict(obs, deterministic=False)
    obs, reward, done, info = env.step(action)
    # env.render(mode="rgb_array")