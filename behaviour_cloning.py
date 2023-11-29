"""This is a simple example demonstrating how to clone the behavior of an expert.

Refer to the jupyter notebooks for more detailed examples of how to use the algorithms.
"""
# import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy

# from imitation.algorithms import bc
# from imitation.util.util import make_vec_env
from gym_data.envs import CarcassoneEnv
# device = torch.device("cuda")
# register(id='Carcassone-v0',entry_point='gym_data.envs:CarcassoneEnv',) 

import numpy as np
from imitation.algorithms import bc
from imitation.data import rollout
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
# Create the Carcassonne environment
env = CarcassoneEnv() # make_vec_env(CarcassoneEnv, seed=42,  n_envs=16, vec_env_cls=SubprocVecEnv)  # Use your Carcassonne environment constructor here



import gym
import torch
import torch.nn as nn

class BehaviorCloningModel(nn.Module):
    def __init__(self, observation_space, action_space):
        super(BehaviorCloningModel, self).__init__()
        self.extractors = {}
        
        for key, subspace in observation_space.spaces.items():
            # print(key, subspace.shape)
            if key == "image":
                # We will just downsample one channel of the image by 4x4 and flatten.
                # Assume the image is single-channel (subspace.shape[0] == 0)
                self.extractors[key] = nn.Sequential(nn.MaxPool2d(2), nn.Flatten())
                total_concat_size += subspace.shape[1] // 2 * subspace.shape[2] // 2
            elif key == "vector":
                # Run through a simple MLP
                self.extractors[key] = nn.Linear(subspace.shape[0], 128)
                total_concat_size += 128
            elif key == "big_farmer_planes":
                self.extractors[key] = nn.Sequential(nn.Conv2d(subspace.shape[0], 32, 5, stride=1), nn.BatchNorm2d(32), nn.Conv2d(32, 64,3, stride=1), nn.BatchNorm2d(64), nn.MaxPool2d(2), 
                                                nn.Flatten()) 
                total_concat_size += 128
            elif key == "big_meeples_planes":
                self.extractors[key] = nn.Sequential(nn.Conv2d(subspace.shape[0], 32, 5, stride=1), nn.BatchNorm2d(32), nn.Conv2d(32, 64,3, stride=1), nn.BatchNorm2d(64), nn.MaxPool2d(2), 
                                                nn.Flatten())  
                total_concat_size += 128
                
            elif key == "chapel_plane":
                self.extractors[key] = nn.Sequential( nn.Flatten(), nn.Linear(subspace.shape[0] * subspace.shape[1] , 512),nn.LeakyReLU(), nn.Linear(512,128),nn.LeakyReLU(), nn.Linear(128,64),nn.LeakyReLU()) 
                total_concat_size += 64
            elif key == "city_planes":
                self.extractors[key] = nn.Sequential(nn.Conv2d(subspace.shape[0], 32, 5, stride=1), nn.BatchNorm2d(32), nn.Conv2d(32, 64,3, stride=1), nn.BatchNorm2d(64), nn.MaxPool2d(2), 
                                                nn.Flatten())  
                total_concat_size += 128
            elif key == "farmer_planes":
                self.extractors[key] = nn.Sequential(nn.Conv2d(subspace.shape[0], 32, 5, stride=1), nn.BatchNorm2d(32), nn.Conv2d(32, 64,3, stride=1), nn.BatchNorm2d(64), nn.MaxPool2d(2), 
                                                nn.Flatten())   
                total_concat_size += 128
            elif key == "field_planes":
                self.extractors[key] = nn.Sequential(nn.Conv2d(subspace.shape[0], 64, 3, stride=1, padding="same"), nn.BatchNorm2d(64), nn.Conv2d(64, 128,3, stride=1), nn.BatchNorm2d(128), nn.MaxPool2d(2),
                                                nn.Conv2d(128, 256,3, stride=1), nn.BatchNorm2d(256), nn.MaxPool2d(2),
                                                nn.Flatten())  
                total_concat_size += 128
            elif key == "flowers_plane":
                self.extractors[key] = nn.Sequential( nn.Flatten(), nn.Linear(subspace.shape[0] * subspace.shape[1] , 512),nn.LeakyReLU(), nn.Linear(512,128),nn.LeakyReLU()) 
                total_concat_size += 64
            elif key == "meeple_planes":
                self.extractors[key] = nn.Sequential(nn.Conv2d(subspace.shape[0], 32, 5, stride=1), nn.BatchNorm2d(32), nn.Conv2d(32, 64,3, stride=1), nn.BatchNorm2d(64), nn.MaxPool2d(2), 
                                                nn.Flatten()) 
                total_concat_size += 128
            elif key == "other_properties_plane":
                # print(subspace.shape)
                self.extractors[key] = nn.Sequential(nn.Linear(611 , 128))
                total_concat_size += 128
            elif key == "road_planes":
                self.extractors[key] = nn.Sequential(nn.Conv2d(subspace.shape[0], 32, 5, stride=1), nn.BatchNorm2d(32), nn.Conv2d(32, 64,3, stride=1), nn.BatchNorm2d(64), nn.MaxPool2d(2), 
                                                nn.Flatten()) 
                total_concat_size += 128
            elif key == "shield_plane":
                self.extractors[key] = nn.Sequential( nn.Flatten(), nn.Linear(subspace.shape[0] * subspace.shape[1] , 512), nn.LeakyReLU(), nn.Linear(512,128), nn.LeakyReLU()) 
                total_concat_size += 64
            elif key == "abbot_planes":
                self.extractors[key] = nn.Sequential(nn.Conv2d(subspace.shape[0], 32, 5, stride=1), nn.BatchNorm2d(32), nn.Conv2d(32, 64,3, stride=1), nn.BatchNorm2d(64), nn.MaxPool2d(2), 
                                                nn.Flatten()) 
                total_concat_size += 64
                
        total_concat_size = 2496

        # Assuming action_space is a gym.spaces.MultiDiscrete
        
        self.fc1 = nn.Linear(
            total_concat_size, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 128)
        
        self.fcs = nn.Sequential(self.fc1,torch.relu(),self.fc2,torch.relu(),self.fc3)
        
        # Assuming action_space is a gym.spaces.MultiDiscrete
        self.discrete_action_branches = nn.ModuleList([
            nn.Sequential(nn.Linear(128, 64),torch.relu(),  nn.Linear(64, n)) for n in action_space.nvec
        ])

    def forward(self, obs):
        preprocessed = []
        
        for key, subspace in obs.spaces.items():
            preprocessed.append(self.extractors[key](obs["key"]))
            

        x = torch.relu(self.fcs(torch.cat(preprocessed, dim=1)))

        # Calculate the logits for each discrete action component
        action_logits = [branch(x) for branch in self.discrete_action_branches]

        return action_logits

n_players = 2

from gymnasium import spaces
board_size = 10
        
other_properties_space = np.ones(12)*2

other_properties_space[-4] = 250
other_properties_space[-3] = 250
other_properties_space[-2] = n_players
other_properties_space[-1] = 3
other_properties_space[0] = 15
other_properties_space[1] = 10
other_properties_space[5] = 59
other_properties_space[6] = 8 # Meeples
other_properties_space[7] = 8

observation_space = spaces.Dict(
{
    "city_planes": spaces.MultiBinary([15, board_size, board_size]),
    "road_planes": spaces.MultiBinary([10, board_size, board_size]),
    "chapel_plane": spaces.MultiBinary([board_size, board_size]),
    "shield_plane": spaces.MultiBinary([board_size, board_size]),
    "flowers_plane": spaces.MultiBinary([board_size, board_size]),
    "field_planes": spaces.MultiBinary([59, board_size, board_size]),
    "meeple_planes": spaces.MultiBinary([5 * n_players, board_size, board_size]),
    "abbot_planes": spaces.MultiBinary([n_players, board_size, board_size]),
    "farmer_planes": spaces.MultiBinary([9 * n_players, board_size, board_size]),
    "big_farmer_planes": spaces.MultiBinary([9 * n_players, board_size, board_size]),
    "big_meeples_planes": spaces.MultiBinary([5 * n_players, board_size, board_size]),
    "other_properties_plane": spaces.MultiDiscrete(other_properties_space)}
)
import torch.optim as optim

action_space = spaces.MultiDiscrete([3, board_size, board_size, 4, 9, 5])
# Create the Behavior Cloning model
behavior_cloning_model = BehaviorCloningModel(observation_space, action_space)

criterion = nn.CrossEntropyLoss()  # Use an appropriate loss function
optimizer = optim.Adam(behavior_cloning_model.parameters(), lr=0.001)

env = CarcassoneEnv()

trajectories = env.generate_trajectory(1e6)
# Trajectories would be an array of trajectory

behavior_cloning_model.fit(trajectories)

# Initialize PPO with the behavior cloning model
ppo_policy = ActorCriticPolicy(
    env.observation_space,
    env.action_space,
    net_arch=[256, 256]  # Adjust architecture as needed
)

ppo_agent = PPO(
    ppo_policy,
    env,
    verbose=1,
    tensorboard_log="./ppo_behavior_cloning_tensorboard/",
    policy_kwargs={"behavior_cloning_model": behavior_cloning_model}
)

# Train the PPO agent further
ppo_agent.learn(total_timesteps=10000)

# Evaluate the trained agent
mean_reward, _ = evaluate_policy(ppo_agent, env, n_eval_episodes=10)

print(f"Mean Reward: {mean_reward}")






# Example training loop (adjust as needed)
for epoch in range(num_epochs):
    for batch in dataloader:  # DataLoader for your example trajectories
        optimizer.zero_grad()

        obs_dict = batch['observations']  # Dictionary of observations
        target_actions = batch['actions']  # Target actions from example trajectories

        action_logits = behavior_cloning_model(obs_dict)
        loss = sum([criterion(logits, targets) for logits, targets in zip(action_logits, target_actions)])

        loss.backward()
        optimizer.step()
        
        







































# Generate trajectories with random actions
rng = np.random.default_rng(0)
num_trajectories = 1  # You can adjust this as needed
rollouts = []

for _ in range(num_trajectories):
    obs = env.reset()
    done = False
    rollout_data = {"obs": [], "acts": [], "rewards": [], "terminal": [], "infos": []}

    while not done:
        action = env.get_random_action_coded()  # Generate random actions
        next_obs, reward, done, _ = env.step(action)

        rollout_data["obs"].append(obs)
        rollout_data["acts"].append(action)
        rollout_data["rewards"].append(reward)
        rollout_data["terminal"].append(done)

        obs = next_obs

    rollouts.append(rollout_data)

# Convert the rollouts to the required format for BC training
transitions = rollout.flatten_trajectories(rollouts)

# Create a BC trainer and train the model
bc_trainer = bc.BC(
    observation_space=env.observation_space,
    action_space=env.action_space,
    demonstrations=transitions,
    rng=rng,
)
bc_trainer.train(n_epochs=1)

# Evaluate the trained policy
reward, _ = evaluate_policy(bc_trainer.policy, env, 10)
print("Reward:", reward)
