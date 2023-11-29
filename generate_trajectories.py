"""This is a simple example demonstrating how to clone the behavior of an expert.

Refer to the jupyter notebooks for more detailed examples of how to use the algorithms.
"""
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
import json
from gym_data.envs import CarcassoneEnv
import numpy as np
from gymnasium import spaces

def numpy_array_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [numpy_array_to_list(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: numpy_array_to_list(value) for key, value in obj.items()}
    else:
        return obj
    
import time  
import gc
from itertools import chain

if __name__ == '__main__':
    
    
    env = make_vec_env(CarcassoneEnv, seed=42,  n_envs=8, vec_env_cls=SubprocVecEnv)
    
    for i in range(8):
        env.env_method("set_random", int(time.time()), indices=[i])
        time.sleep(1)
    
    # trajectories = list(chain.from_iterable(env.env_method("generate_trajectory", 100)))

    trajectories = env.env_method("generate_trajectory", 10000)
    
    env.close()
    
    env = None
    
    gc.collect()
    
    print("Trajectories generated. Reformating")
    
    trajectories = list(chain.from_iterable(trajectories)) #[item for sublist in trajectories for item in sublist]

    with open("trajectories.json", "w") as json_file:
        json.dump(numpy_array_to_list(trajectories), json_file)

   