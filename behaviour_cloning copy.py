"""This is a simple example demonstrating how to clone the behavior of an expert.

Refer to the jupyter notebooks for more detailed examples of how to use the algorithms.
"""
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
import json
from gym_data.envs import CarcassoneEnv
import numpy as np
import torch
import torch.nn as nn
# from torchinfo import summary

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
if __name__ == '__main__':
    total_concat_size = 2468
    # class BehaviorCloningModel(nn.Module):
    #     def __init__(self, observation_space, action_space):
    #         super(BehaviorCloningModel, self).__init__()
    #         self.extractors = {}
    #         total_concat_size = 2468
            
    #         for key, subspace in observation_space.spaces.items():
    #             # print(key, subspace.shape)
    #             if key == "image":
    #                 # We will just downsample one channel of the image by 4x4 and flatten.
    #                 # Assume the image is single-channel (subspace.shape[0] == 0)
    #                 self.extractors[key] = nn.Sequential(nn.MaxPool2d(2), nn.Flatten())
    #                 total_concat_size += subspace.shape[1] // 2 * subspace.shape[2] // 2
    #             elif key == "vector":
    #                 # Run through a simple MLP
    #                 self.extractors[key] = nn.Linear(subspace.shape[0], 128)
    #                 total_concat_size += 128
    #             elif key == "big_farmer_planes":
    #                 self.extractors[key] = nn.Sequential(nn.Conv2d(subspace.shape[0], 64, 5, stride=1, padding="same"), nn.BatchNorm2d(64), nn.Conv2d(64, 128,3, stride=1, padding="same"), nn.BatchNorm2d(128), nn.MaxPool2d(2), nn.Conv2d(128, 256,3, stride=1, padding="same"), nn.BatchNorm2d(256), nn.MaxPool2d(2), 
    #                                                 nn.Flatten()) 
    #                 total_concat_size += 128
    #             elif key == "big_meeples_planes":
    #                 self.extractors[key] = nn.Sequential(nn.Conv2d(subspace.shape[0], 64, 5, stride=1, padding="same"), nn.BatchNorm2d(64), nn.Conv2d(64, 128,3, stride=1, padding="same"), nn.BatchNorm2d(128), nn.MaxPool2d(2), nn.Conv2d(128, 256,3, stride=1, padding="same"), nn.BatchNorm2d(256), nn.MaxPool2d(2), 
    #                                                 nn.Flatten())  
    #                 total_concat_size += 128
                    
    #             elif key == "chapel_plane":
    #                 self.extractors[key] = nn.Sequential( nn.Flatten(), nn.Linear(subspace.shape[0] * subspace.shape[1] , 512),nn.LeakyReLU(), nn.Linear(512,128),nn.LeakyReLU(), nn.Linear(128,64),nn.LeakyReLU()) 
    #                 total_concat_size += 64
    #             elif key == "city_planes":
    #                 self.extractors[key] = nn.Sequential(nn.Conv2d(subspace.shape[0], 64, 5, stride=1, padding="same"), nn.BatchNorm2d(64), nn.Conv2d(64, 128,3, stride=1, padding="same"), nn.BatchNorm2d(128), nn.MaxPool2d(2), nn.Conv2d(128, 256,3, stride=1, padding="same"), nn.BatchNorm2d(256), nn.MaxPool2d(2), 
    #                                                 nn.Flatten())  
    #                 total_concat_size += 128
    #             elif key == "farmer_planes":
    #                 self.extractors[key] = nn.Sequential(nn.Conv2d(subspace.shape[0], 64, 5, stride=1, padding="same"), nn.BatchNorm2d(64), nn.Conv2d(64, 128,3, stride=1, padding="same"), nn.BatchNorm2d(128), nn.MaxPool2d(2), nn.Conv2d(128, 256,3, stride=1, padding="same"), nn.BatchNorm2d(256), nn.MaxPool2d(2), 
    #                                                 nn.Flatten()) 
    #                 total_concat_size += 128
    #             elif key == "field_planes":
    #                 self.extractors[key] = nn.Sequential(nn.Conv2d(subspace.shape[0], 128, 3, stride=1, padding="same"), nn.BatchNorm2d(128), nn.Conv2d(128, 256,3, stride=1, padding="same"), nn.BatchNorm2d(256), nn.MaxPool2d(2),
    #                                                 nn.Conv2d(256, 512,3, stride=1, padding="same"), nn.BatchNorm2d(512), nn.MaxPool2d(2),nn.Conv2d(512, 512,3, stride=1, padding="same"), nn.BatchNorm2d(512),
    #                                                 nn.Flatten())  
    #                 total_concat_size += 128
    #             elif key == "flowers_plane":
    #                 self.extractors[key] = nn.Sequential( nn.Flatten(), nn.Linear(subspace.shape[0] * subspace.shape[1] , 512),nn.LeakyReLU(), nn.Linear(512,128),nn.LeakyReLU()) 
    #                 total_concat_size += 64
    #             elif key == "meeple_planes":
    #                 self.extractors[key] = nn.Sequential(nn.Conv2d(subspace.shape[0], 64, 5, stride=1, padding="same"), nn.BatchNorm2d(64), nn.Conv2d(64, 128,3, stride=1, padding="same"), nn.BatchNorm2d(128), nn.MaxPool2d(2), nn.Conv2d(128, 256,3, stride=1, padding="same"), nn.BatchNorm2d(256), nn.MaxPool2d(2), 
    #                                                 nn.Flatten()) 
    #                 total_concat_size += 128
    #             elif key == "other_properties_plane":
    #                 # print(subspace.shape)
    #                 self.extractors[key] = nn.Sequential(nn.Linear(12 , 128))
    #                 total_concat_size += 128
    #             elif key == "road_planes":
    #                 self.extractors[key] = nn.Sequential(nn.Conv2d(subspace.shape[0], 64, 5, stride=1, padding="same"), nn.BatchNorm2d(64), nn.Conv2d(64, 128,3, stride=1, padding="same"), nn.BatchNorm2d(128), nn.MaxPool2d(2), nn.Conv2d(128, 256,3, stride=1, padding="same"), nn.BatchNorm2d(256), nn.MaxPool2d(2), 
    #                                                 nn.Flatten())  
    #                 total_concat_size += 128
    #             elif key == "shield_plane":
    #                 self.extractors[key] = nn.Sequential( nn.Flatten(), nn.Linear(subspace.shape[0] * subspace.shape[1] , 512), nn.LeakyReLU(), nn.Linear(512,128), nn.LeakyReLU()) 
    #                 total_concat_size += 64
    #             elif key == "abbot_planes":
    #                 self.extractors[key] = nn.Sequential(nn.Conv2d(subspace.shape[0], 64, 5, stride=1, padding="same"), nn.BatchNorm2d(64), nn.Conv2d(64, 128,3, stride=1, padding="same"), nn.BatchNorm2d(128), nn.MaxPool2d(2), nn.Conv2d(128, 256,3, stride=1, padding="same"), nn.BatchNorm2d(256), nn.MaxPool2d(2), 
    #                                                 nn.Flatten()) 
    #                 total_concat_size += 64
                    
    #         total_concat_size = 2496

    #         # Assuming action_space is a gym.spaces.MultiDiscrete
            
    #         self.fc1 = nn.Linear(
    #             total_concat_size, 1024)
    #         self.fc2 = nn.Linear(1024, 256)
    #         self.fc3 = nn.Linear(256, 128)
            
    #         self.fcs = nn.Sequential(self.fc1,nn.LeakyReLU(),self.fc2,nn.LeakyReLU(),self.fc3)
            
    #         # Assuming action_space is a gym.spaces.MultiDiscrete
    #         self.discrete_action_branches = nn.ModuleList([
    #             nn.Sequential(nn.Linear(128, 64),nn.LeakyReLU(),  nn.Linear(64, n), nn.Softmax()) for n in action_space.nvec
    #         ])

    #     def forward(self, obs_batch):
    #         actions = []
    #         for obs in obs_batch:
    #             obs_pre = []
    #             for key, subspace in obs.items():
    #                 input_extractor = torch.tensor(obs[key], dtype=torch.float32).unsqueeze(0)
    #                 obs_pre.append(self.extractors[key](input_extractor))
                    
    #             x = torch.relu(self.fcs(torch.cat(obs_pre, dim =1)))

    #             # Calculate the logits for each discrete action component
    #             action_logits = [branch(x) for branch in self.discrete_action_branches]
    #             actions.append(action_logits)
                
    #         return actions

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
    # behavior_cloning_model = BehaviorCloningModel(observation_space, action_space)

    # criterion = nn.CrossEntropyLoss()  # Use an appropriate loss function
    # optimizer = optim.Adam(behavior_cloning_model.parameters(), lr=0.001)
    env = make_vec_env(CarcassoneEnv, seed=42,  n_envs=16, vec_env_cls=SubprocVecEnv)
    for i in range(16):
        # Here, we call a hypothetical method "my_custom_method" with different arguments for each environment
        env.env_method("set_random", int(time.time()), indices=[i])
        time.sleep(1)
        
    trajectories = env.env_method("generate_trajectory", 1000)
    
    trajectories = [item for sublist in trajectories for item in sublist]

    # with open("trajectories.json", "w") as json_file:
    #     json_traj = numpy_array_to_list(trajectories)
    #     print(json_traj[0])
    #     json.dump(json_traj, json_file)
    
    # Trajectories would be an array of trajectory


    # num_epochs = 100
    # num_steps_per_epoch = 100
    # batch_size = 32

    def one_hot(value, max):
        res = np.zeros(max)
        res[value] = 1
        return res


    # for epoch in range(num_epochs):
    #     for _ in range(num_steps_per_epoch // batch_size):
    #         batch_indices = torch.randint(0, len(trajectories), (batch_size,))
    #         batch = [trajectories[i] for i in batch_indices]
    #         # Extract states, actions, rewards, etc. from the batch
    #         states = [batch[i]["state"] for i in range(len(batch))]
    #         actions = [batch[i]["action"] for i in range(len(batch))] # actions is a list of lists that represent the action to be taken by each batch
            
    #         outputs = behavior_cloning_model(states)  # outputs is a 32 length list that conatins the list of probabilities to select one move
    #         new_actions = []
    #         for a in actions:
    #             new_a = []
    #             new_a.append(one_hot(a[0],3))
    #             new_a.append(one_hot(a[1],board_size))
    #             new_a.append(one_hot(a[2],board_size))
    #             new_a.append(one_hot(a[3],4))
    #             new_a.append(one_hot(a[4],9))
    #             new_a.append(one_hot(a[5],5))
    #             new_actions.append(new_a)
            
    #         # print(outputs)
    #         # print(new_actions)
    #         # loss = criterion(outputs, torch.tensor(actions, dtype=torch.float32))  # Replace 'targets' with your target data
    #         losses = []
    #         for output, target in zip(outputs, new_actions):
    #             for o, t in zip(output, target):
    #                 t = torch.tensor(t, dtype=torch.float32).unsqueeze(0)
    #                 losses.append(criterion(o, t))
                    
    #         average_loss = sum(losses) / len(losses)
    #         # Backpropagation and optimization
    #         optimizer.zero_grad()
    #         average_loss.backward()
    #         optimizer.step()
        
        
    def linear_schedule(initial_value):
        """
        Linear learning rate schedule.
        :param initial_value: (float or str)
        :return: (function)
        """
        if isinstance(initial_value, str):
            initial_value = float(initial_value)

        def func(progress):
            """
            Progress will decrease from 1 (beginning) to 0
            :param progress: (float)
            :return: (float)
            """
            return progress * initial_value

        return func
        

    import torch as th
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

    extractable = ["image", "big_farmer_planes", "big_meeples_planes","chapel_plane", "city_planes", "farmer_planes", "field_planes", 
                "flowers_plane", "meeple_planes", "other_properties_plane", "road_planes", "shield_plane", "abbot_planes"]
    class CustomCombinedExtractor(BaseFeaturesExtractor):
        def __init__(self, observation_space: spaces.Dict):
            # We do not know features-dim here before going over all the items,
            # so put something dummy for now. PyTorch requires calling
            # nn.Module.__init__ before adding modules
            # print(observation_space)
            super().__init__(observation_space, features_dim=1)

            extractors = {}

            total_concat_size = 128
            # We need to know size of the output of this extractor,
            # so go over all the spaces and compute output feature sizes
            for key, subspace in observation_space.spaces.items():
                # print(key, subspace.shape)
                if key == "image":
                    # We will just downsample one channel of the image by 4x4 and flatten.
                    # Assume the image is single-channel (subspace.shape[0] == 0)
                    extractors[key] = nn.Sequential(nn.MaxPool2d(2), nn.Flatten())
                    total_concat_size += subspace.shape[1] // 2 * subspace.shape[2] // 2
                elif key == "vector":
                    # Run through a simple MLP
                    extractors[key] = nn.Linear(subspace.shape[0], 128)
                    total_concat_size += 128
                elif key == "big_farmer_planes":
                    extractors[key] = nn.Sequential(nn.Conv2d(subspace.shape[0], 64, 5, stride=1, padding="same"), nn.BatchNorm2d(64), nn.Conv2d(64, 128,3, stride=1, padding="same"), nn.BatchNorm2d(128), nn.MaxPool2d(2), nn.Conv2d(128, 256,3, stride=1, padding="same"), nn.BatchNorm2d(256), nn.MaxPool2d(2), 
                                                    nn.Flatten()) 
                    # print(key)
                    # summary(extractors[key], (1,) + subspace.shape)
                    total_concat_size += 128
                elif key == "big_meeples_planes":
                    extractors[key] = nn.Sequential(nn.Conv2d(subspace.shape[0], 64, 5, stride=1, padding="same"), nn.BatchNorm2d(64), nn.Conv2d(64, 128,3, stride=1, padding="same"), nn.BatchNorm2d(128), nn.MaxPool2d(2), nn.Conv2d(128, 256,3, stride=1, padding="same"), nn.BatchNorm2d(256), nn.MaxPool2d(2), 
                                                    nn.Flatten())  
                    # print(key)
                    # summary(extractors[key], (1,) + subspace.shape)
                    total_concat_size += 128
                    
                elif key == "chapel_plane":
                    extractors[key] = nn.Sequential( nn.Flatten(), nn.Linear(subspace.shape[0] * subspace.shape[1] , 512),nn.LeakyReLU(), nn.Linear(512,128),nn.LeakyReLU(), nn.Linear(128,64),nn.LeakyReLU()) 
                    # print(key)
                    # summary(extractors[key], (1,) + subspace.shape)
                    total_concat_size += 64
                elif key == "city_planes":
                    extractors[key] = nn.Sequential(nn.Conv2d(subspace.shape[0], 64, 5, stride=1, padding="same"), nn.BatchNorm2d(64), nn.Conv2d(64, 128,3, stride=1, padding="same"), nn.BatchNorm2d(128), nn.MaxPool2d(2), nn.Conv2d(128, 256,3, stride=1, padding="same"), nn.BatchNorm2d(256), nn.MaxPool2d(2),
                                                    nn.Flatten())  
                    # print(key)
                    # summary(extractors[key], (1,) + subspace.shape)
                    total_concat_size += 128
                elif key == "farmer_planes":
                    extractors[key] = nn.Sequential(nn.Conv2d(subspace.shape[0], 64, 5, stride=1, padding="same"), nn.BatchNorm2d(64), nn.Conv2d(64, 128,3, stride=1, padding="same"), nn.BatchNorm2d(128), nn.MaxPool2d(2), nn.Conv2d(128, 256,3, stride=1, padding="same"), nn.BatchNorm2d(256), nn.MaxPool2d(2),
                                                    nn.Flatten()) 
                    # print(key)
                    # summary(extractors[key], (1,) + subspace.shape)
                    total_concat_size += 128
                elif key == "field_planes":
                    extractors[key] = nn.Sequential(nn.Conv2d(subspace.shape[0], 128, 3, stride=1, padding="same"), nn.BatchNorm2d(128), nn.Conv2d(128, 256,3, stride=1, padding="same"), nn.BatchNorm2d(256), nn.MaxPool2d(2),
                                                    nn.Conv2d(256, 256,3, stride=1, padding="same"), nn.BatchNorm2d(256), nn.MaxPool2d(2),nn.Conv2d(256, 256,3, stride=1, padding="same"), nn.BatchNorm2d(256),
                                                    nn.Flatten())  
                    # print(key)
                    # summary(extractors[key], (1,) + subspace.shape)
                    total_concat_size += 128
                elif key == "flowers_plane":
                    extractors[key] = nn.Sequential( nn.Flatten(), nn.Linear(subspace.shape[0] * subspace.shape[1] , 512),nn.LeakyReLU(), nn.Linear(512,128),nn.LeakyReLU()) 
                    # print(key)
                    # summary(extractors[key], (1,) + subspace.shape)
                    total_concat_size += 64
                elif key == "meeple_planes":
                    extractors[key] = nn.Sequential(nn.Conv2d(subspace.shape[0], 64, 5, stride=1, padding="same"), nn.BatchNorm2d(64), nn.Conv2d(64, 128,3, stride=1, padding="same"), nn.BatchNorm2d(128), nn.MaxPool2d(2), nn.Conv2d(128, 256,3, stride=1, padding="same"), nn.BatchNorm2d(256), nn.MaxPool2d(2), 
                                                    nn.Flatten()) 
                    # print(key)
                    # summary(extractors[key], (1,) + subspace.shape)
                    total_concat_size += 128
                elif key == "other_properties_plane":
                    # print(subspace.shape)
                    extractors[key] = nn.Sequential(nn.Linear(618 , 128))
                    # print(key)
                    # summary(extractors[key], (1,) + subspace.shape)
                    total_concat_size += 128
                elif key == "road_planes":
                    extractors[key] = nn.Sequential(nn.Conv2d(subspace.shape[0], 64, 5, stride=1, padding="same"), nn.BatchNorm2d(64), nn.Conv2d(64, 128,3, stride=1, padding="same"), nn.BatchNorm2d(128), nn.MaxPool2d(2), nn.Conv2d(128, 256,3, stride=1, padding="same"), nn.BatchNorm2d(256), nn.MaxPool2d(2),
                                                    nn.Flatten())  
                    # print(key)
                    # summary(extractors[key], (1,) + subspace.shape)
                    total_concat_size += 128
                elif key == "shield_plane":
                    extractors[key] = nn.Sequential( nn.Flatten(), nn.Linear(subspace.shape[0] * subspace.shape[1] , 512), nn.LeakyReLU(), nn.Linear(512,128), nn.LeakyReLU()) 
                    # print(key)
                    # summary(extractors[key], (1,) + subspace.shape)
                    total_concat_size += 64
                elif key == "abbot_planes":
                    extractors[key] = nn.Sequential(nn.Conv2d(subspace.shape[0], 64, 5, stride=1, padding="same"), nn.BatchNorm2d(64), nn.Conv2d(64, 128,3, stride=1, padding="same"), nn.BatchNorm2d(128), nn.MaxPool2d(2), nn.Conv2d(128, 256,3, stride=1, padding="same"), nn.BatchNorm2d(256), nn.MaxPool2d(2), 
                                                    nn.Flatten()) 
                    # print(key, subspace.shape)
                    # summary(extractors[key], (1,) + subspace.shape)
                    
                    total_concat_size += 64
                    
            total_concat_size = 8640
            self.extractors = nn.ModuleDict(extractors)
            print("Total concat size: " , total_concat_size)
            # Update the features dim manually
            self._features_dim = total_concat_size
    #         print(total_concat_size)
        def forward(self, observations) -> th.Tensor:
            encoded_tensor_list = []

            # self.extractors contain nn.Modules that do all the processing.
            for key, extractor in self.extractors.items():
                # print(key)
                # print(observations[key].shape)
                extractor_value = extractor(observations[key])
                # if key == "other_properties_plane":
                    # print("Extractor value: ",  extractor_value.shape)
                encoded_tensor_list.append(extractor_value)
            # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
            concated = th.cat(encoded_tensor_list, dim=1)
            # print("Forward total concat size: " , concated.shape)

            return concated
        
    from stable_baselines3 import PPO, A2C
        
         
    policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractor,
        net_arch=[dict(pi=[8192, 8192, 4096],
                       vf=[4096, 1024, 256, 128, 128])]
    )

    # env = make_vec_env(CarcassoneEnv, seed=42,  n_envs=4, vec_env_cls=SubprocVecEnv)

    from torch.utils.data.dataset import Dataset, random_split


    class ExpertDataSet(Dataset):
        def __init__(self, expert_observations, expert_actions):
            self.observations = expert_observations
            self.actions = expert_actions

        def __getitem__(self, index):
            return (self.observations[index], self.actions[index])

        def __len__(self):
            return len(self.observations)
        
    states = [trajectories[i]["state"] for i in range(len(trajectories))]
    actions = [trajectories[i]["action"] for i in range(len(trajectories))]

    expert_dataset = ExpertDataSet(states, actions)

    train_size = int(0.8 * len(expert_dataset))

    test_size = len(expert_dataset) - train_size

    train_expert_dataset, test_expert_dataset = random_split(
        expert_dataset, [train_size, test_size]
    )
    print("trajectories generated")
    from torch.optim.lr_scheduler import StepLR

    a2c_student = A2C("MultiInputPolicy", env, verbose=1, policy_kwargs=policy_kwargs)

    def pretrain_agent(
        student,
        batch_size=64,
        epochs=1000,
        scheduler_gamma=0.7,
        learning_rate=1.0,
        log_interval=100,
        no_cuda=True,
        seed=1,
        test_batch_size=64,
    ):
        use_cuda = not no_cuda and th.cuda.is_available()
        th.manual_seed(seed)
        device = th.device("cuda" if use_cuda else "cpu")
        kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

        if isinstance(env.action_space, spaces.Box):
            criterion = nn.MSELoss()
        else:
            criterion = nn.CrossEntropyLoss()

        # Extract initial policy
        model = student.policy.to(device)

        def train(model, device, train_loader, optimizer):
            model.train()

            for batch_idx, (data, target) in enumerate(train_loader):
                for key, value in data.items():
                    data[key] = data[key].to(device)
                target = target.to(device)
                optimizer.zero_grad()

                if isinstance(env.action_space, spaces.Box):
                    # A2C/PPO policy outputs actions, values, log_prob
                    # SAC/TD3 policy outputs actions only
                    if isinstance(student, (A2C, PPO)):
                        action, _, _ = model(data)
                    else:
                        # SAC/TD3:
                        action = model(data)
                    action_prediction = action.double()
                else:
                    # Retrieve the logits for A2C/PPO when using discrete actions
                    dist = model.get_distribution(data)
                    action_prediction = [dist_logs.logits for dist_logs in dist.distribution]
                    target = target.long()

                losses = []
                new_actions = []
                aa = []
                ab = []
                ac = []
                ad = []
                ae = []
                af = []
                for a in target:
                    aa.append(one_hot(a[0],3))
                    ab.append(one_hot(a[1],board_size))
                    ac.append(one_hot(a[2],board_size))
                    ad.append(one_hot(a[3],4))
                    ae.append(one_hot(a[4],9))
                    af.append(one_hot(a[5],5))
                new_actions.append(aa)
                new_actions.append(ab)
                new_actions.append(ac)
                new_actions.append(ad)
                new_actions.append(ae)
                new_actions.append(af)
                    
                target = new_actions
                for output, targets in zip(action_prediction, target):
                    for o, t in zip(output, targets):
                        t = torch.tensor(t, dtype=torch.float32).unsqueeze(0).view(-1)
                        losses.append(criterion(o, t))
                        
                average_loss = sum(losses) / len(losses)
                # loss = criterion(action_prediction, target)
                average_loss.backward()
                optimizer.step()
                if batch_idx % log_interval == 0:
                    print(
                        "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                            epoch,
                            batch_idx * len(data),
                            len(train_loader.dataset),
                            100.0 * batch_idx / len(train_loader),
                            average_loss.item(),
                        )
                    )

        def test(model, device, test_loader):
            model.eval()
            test_loss = 0
            with th.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)

                    if isinstance(env.action_space, spaces.Box):
                        # A2C/PPO policy outputs actions, values, log_prob
                        # SAC/TD3 policy outputs actions only
                        if isinstance(student, (A2C, PPO)):
                            action, _, _ = model(data)
                        else:
                            # SAC/TD3:
                            action = model(data)
                        action_prediction = action.double()
                    else:
                        # Retrieve the logits for A2C/PPO when using discrete actions
                        dist = model.get_distribution(data)
                        action_prediction = dist.distribution.logits
                        target = target.long()

                    test_loss = criterion(action_prediction, target)
            test_loss /= len(test_loader.dataset)
            print(f"Test set: Average loss: {test_loss:.4f}")

        # Here, we use PyTorch `DataLoader` to our load previously created `ExpertDataset` for training
        # and testing
        train_loader = th.utils.data.DataLoader(
            dataset=train_expert_dataset, batch_size=batch_size, shuffle=True, **kwargs
        )
        test_loader = th.utils.data.DataLoader(
            dataset=test_expert_dataset,
            batch_size=test_batch_size,
            shuffle=True,
            **kwargs,
        )

        # Define an Optimizer and a learning rate schedule.
        optimizer = optim.Adadelta(model.parameters(), lr=learning_rate)
        scheduler = StepLR(optimizer, step_size=1, gamma=scheduler_gamma)

        # Now we are finally ready to train the policy model.
        for epoch in range(1, epochs + 1):
            train(model, device, train_loader, optimizer)
            # test(model, device, test_loader)
            scheduler.step()

        # Implant the trained policy network back into the RL student agent
        a2c_student.policy = model
        
    pretrain_agent(
        a2c_student,
        epochs=100,
        scheduler_gamma=0.7,
        learning_rate=1.0,
        log_interval=100,
        no_cuda=True,
        seed=1,
        batch_size=256,
        test_batch_size=64,
    )

    mean_reward, _ = evaluate_policy(a2c_student, env, n_eval_episodes=10)

    print(f"Mean Reward: {mean_reward}")
    
    # a2c_student.policy = 
    a2c_student.policy.to("cuda")

    a2c_student.learn(100000)

    mean_reward, _ = evaluate_policy(a2c_student, env, n_eval_episodes=10)

    print(f"Mean Reward: {mean_reward}")


    # print("ppo creation")
    # ppo_agent = PPO(ppo_policy, env,verbose=1, batch_size=64,tensorboard_log="runs/",device="auto", n_steps=1024, learning_rate=linear_schedule(0.0003), policy_kwargs=policy_kwargs)

    # print("learning")
    # # Train the PPO agent further
    # ppo_agent.learn(total_timesteps=2000)

    # print("evaluation")
    # # Evaluate the trained agent

