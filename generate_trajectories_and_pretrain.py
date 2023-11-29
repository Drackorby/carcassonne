"""This is a simple example demonstrating how to clone the behavior of an expert.

Refer to the jupyter notebooks for more detailed examples of how to use the algorithms.
"""
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
import json
from gym_data.envs import CarcassoneEnv
import numpy as np
from gymnasium import spaces
import time
import gc
from itertools import chain
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
import json
from gym_data.envs import CarcassoneEnv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch as th
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import PPO, A2C
from torch.utils.data.dataset import Dataset, random_split
from torch.optim.lr_scheduler import StepLR

extractable = ["image", "big_farmer_planes", "big_meeples_planes", "chapel_plane", "city_planes", "farmer_planes",
                   "field_planes",
                   "flowers_plane", "meeple_planes", "other_properties_plane", "road_planes", "shield_plane",
                   "abbot_planes"]
class CustomCombinedExtractorOld(BaseFeaturesExtractor):
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
                extractors[key] = nn.Sequential(nn.Conv2d(subspace.shape[0], 64, 5, stride=1, padding="same"),
                                                nn.BatchNorm2d(64), nn.Conv2d(64, 128, 3, stride=1, padding="same"),
                                                nn.BatchNorm2d(128), nn.MaxPool2d(2),
                                                nn.Conv2d(128, 256, 3, stride=1, padding="same"),
                                                nn.BatchNorm2d(256), nn.MaxPool2d(2),
                                                nn.Flatten())
                total_concat_size += 128
            elif key == "big_meeples_planes":
                extractors[key] = nn.Sequential(nn.Conv2d(subspace.shape[0], 64, 5, stride=1, padding="same"),
                                                nn.BatchNorm2d(64), nn.Conv2d(64, 128, 3, stride=1, padding="same"),
                                                nn.BatchNorm2d(128), nn.MaxPool2d(2),
                                                nn.Conv2d(128, 256, 3, stride=1, padding="same"),
                                                nn.BatchNorm2d(256), nn.MaxPool2d(2),
                                                nn.Flatten())
                total_concat_size += 128

            elif key == "chapel_plane":
                extractors[key] = nn.Sequential(nn.Flatten(), nn.Linear(subspace.shape[0] * subspace.shape[1], 512),
                                                nn.LeakyReLU(), nn.Linear(512, 128), nn.LeakyReLU(),
                                                nn.Linear(128, 64), nn.LeakyReLU())
                total_concat_size += 64
            elif key == "city_planes":
                extractors[key] = nn.Sequential(nn.Conv2d(subspace.shape[0], 64, 5, stride=1, padding="same"),
                                                nn.BatchNorm2d(64), nn.Conv2d(64, 128, 3, stride=1, padding="same"),
                                                nn.BatchNorm2d(128), nn.MaxPool2d(2),
                                                nn.Conv2d(128, 256, 3, stride=1, padding="same"),
                                                nn.BatchNorm2d(256), nn.MaxPool2d(2),
                                                nn.Conv2d(256, 512, 3, stride=1, padding="same"),
                                                nn.Flatten())
                total_concat_size += 128
            elif key == "farmer_planes":
                extractors[key] = nn.Sequential(nn.Conv2d(subspace.shape[0], 64, 5, stride=1, padding="same"),
                                                nn.BatchNorm2d(64), nn.Conv2d(64, 128, 3, stride=1, padding="same"),
                                                nn.BatchNorm2d(128), nn.MaxPool2d(2),
                                                nn.Conv2d(128, 256, 3, stride=1, padding="same"),
                                                nn.BatchNorm2d(256), nn.MaxPool2d(2),
                                                nn.Flatten())
                total_concat_size += 128
            elif key == "field_planes":
                extractors[key] = nn.Sequential(nn.Conv2d(subspace.shape[0], 128, 3, stride=1, padding="same"),
                                                nn.BatchNorm2d(128),
                                                nn.Conv2d(128, 256, 3, stride=1, padding="same"),
                                                nn.BatchNorm2d(256), nn.MaxPool2d(2),
                                                nn.Conv2d(256, 512, 3, stride=1, padding="same"),
                                                nn.BatchNorm2d(512), nn.MaxPool2d(2),
                                                nn.Conv2d(512, 512, 3, stride=1, padding="same"),
                                                nn.BatchNorm2d(512),
                                                nn.Flatten())
                total_concat_size += 128
            elif key == "flowers_plane":
                extractors[key] = nn.Sequential(nn.Flatten(), nn.Linear(subspace.shape[0] * subspace.shape[1], 512),
                                                nn.LeakyReLU(), nn.Linear(512, 128), nn.LeakyReLU())
                total_concat_size += 64
            elif key == "meeple_planes":
                extractors[key] = nn.Sequential(nn.Conv2d(subspace.shape[0], 64, 5, stride=1, padding="same"),
                                                nn.BatchNorm2d(64), nn.Conv2d(64, 128, 3, stride=1, padding="same"),
                                                nn.BatchNorm2d(128), nn.MaxPool2d(2),
                                                nn.Conv2d(128, 256, 3, stride=1, padding="same"),
                                                nn.BatchNorm2d(256), nn.MaxPool2d(2),
                                                nn.Flatten())
                total_concat_size += 128
            elif key == "other_properties_plane":
                # print(subspace.shape)
                extractors[key] = nn.Sequential(nn.Linear(618, 512), nn.LeakyReLU(), nn.Linear(512, 256))
                # print(key)
                # summary(extractors[key], (1,) + subspace.shape)
                total_concat_size += 128
            elif key == "road_planes":
                extractors[key] = nn.Sequential(nn.Conv2d(subspace.shape[0], 64, 5, stride=1, padding="same"),
                                                nn.BatchNorm2d(64), nn.Conv2d(64, 128, 3, stride=1, padding="same"),
                                                nn.BatchNorm2d(128), nn.MaxPool2d(2),
                                                nn.Conv2d(128, 256, 3, stride=1, padding="same"),
                                                nn.BatchNorm2d(256), nn.MaxPool2d(2),
                                                nn.Conv2d(256, 256, 3, stride=1, padding="same"),
                                                nn.Flatten())
                # print(key)
                # summary(extractors[key], (1,) + subspace.shape)
                total_concat_size += 128
            elif key == "shield_plane":
                extractors[key] = nn.Sequential(nn.Flatten(), nn.Linear(subspace.shape[0] * subspace.shape[1], 512),
                                                nn.LeakyReLU(), nn.Linear(512, 128), nn.LeakyReLU())
                # print(key)
                # summary(extractors[key], (1,) + subspace.shape)
                total_concat_size += 64
            elif key == "abbot_planes":
                extractors[key] = nn.Sequential(nn.Conv2d(subspace.shape[0], 64, 5, stride=1, padding="same"),
                                                nn.BatchNorm2d(64), nn.Conv2d(64, 128, 3, stride=1, padding="same"),
                                                nn.BatchNorm2d(128), nn.MaxPool2d(2),
                                                nn.Conv2d(128, 128, 3, stride=1, padding="same"),
                                                nn.BatchNorm2d(128), nn.MaxPool2d(2),
                                                nn.Flatten())
                # print(key, subspace.shape)
                # summary(extractors[key], (1,) + subspace.shape)

                total_concat_size += 64

        total_concat_size = 10304
        self.extractors = nn.ModuleDict(extractors)
        print("Total concat size: ", total_concat_size)
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
                extractors[key] = nn.Sequential(nn.Conv2d(subspace.shape[0], 64, 5, stride=1, padding="same"),
                                                nn.MaxPool2d(2),nn.BatchNorm2d(64), nn.Conv2d(64, 128, 3, stride=1, padding="same"),
                                                nn.MaxPool2d(2),nn.BatchNorm2d(128), 
                                                nn.Conv2d(128, 256, 3, stride=1, padding="same"),
                                                nn.Flatten())
                total_concat_size += 128
            elif key == "big_meeples_planes":
                extractors[key] = nn.Sequential(nn.Conv2d(subspace.shape[0], 64, 5, stride=1, padding="same"),
                                                nn.MaxPool2d(2), nn.BatchNorm2d(64), nn.Conv2d(64, 128, 3, stride=1, padding="same"),
                                                nn.MaxPool2d(2), nn.BatchNorm2d(128), 
                                                nn.Conv2d(128, 256, 3, stride=1, padding="same"),
                                                nn.Flatten())
                total_concat_size += 128

            elif key == "chapel_plane":
                extractors[key] = nn.Sequential(nn.Flatten(), nn.Linear(subspace.shape[0] * subspace.shape[1], 256),
                                                nn.LeakyReLU(), nn.Linear(256, 128), nn.LeakyReLU(),
                                                nn.Linear(128, 64), nn.LeakyReLU())
                total_concat_size += 64
            elif key == "city_planes":
                extractors[key] = nn.Sequential(nn.Conv2d(subspace.shape[0], 64, 5, stride=1, padding="same"),
                                                nn.BatchNorm2d(64), nn.Conv2d(64, 64, 3, stride=1, padding="same"),
                                                nn.MaxPool2d(2), nn.BatchNorm2d(64), 
                                                nn.Conv2d(64, 128, 3, stride=1, padding="same"),
                                                nn.MaxPool2d(2), nn.BatchNorm2d(128),
                                                nn.Conv2d(128, 256, 3, stride=1, padding="same"),
                                                nn.Flatten())
                total_concat_size += 128
            elif key == "farmer_planes":
                extractors[key] = nn.Sequential(nn.Conv2d(subspace.shape[0], 64, 5, stride=1, padding="same"),
                                                nn.MaxPool2d(2),
                                                nn.BatchNorm2d(64), nn.Conv2d(64, 128, 3, stride=1, padding="same"),
                                                nn.MaxPool2d(2),nn.BatchNorm2d(128), 
                                                nn.Conv2d(128, 256, 3, stride=1, padding="same"),
                                                nn.Flatten())
                total_concat_size += 128
            elif key == "field_planes":
                extractors[key] = nn.Sequential(nn.Conv2d(subspace.shape[0], 128, 3, stride=1, padding="same"),
                                                nn.MaxPool2d(2),
                                                nn.BatchNorm2d(128),
                                                nn.Conv2d(128, 128, 3, stride=1, padding="same"),
                                                nn.BatchNorm2d(128), 
                                                nn.Conv2d(128, 256, 3, stride=1, padding="same"),
                                                nn.MaxPool2d(2),
                                                nn.BatchNorm2d(256), 
                                                nn.Conv2d(256, 512, 3, stride=1, padding="same"),
                                                nn.Flatten())
                total_concat_size += 128
            elif key == "flowers_plane":
                extractors[key] = nn.Sequential(nn.Flatten(), nn.Linear(subspace.shape[0] * subspace.shape[1], 512),
                                                nn.LeakyReLU(), nn.Linear(512, 128), nn.LeakyReLU())
                total_concat_size += 64
            elif key == "meeple_planes":
                extractors[key] = nn.Sequential(nn.Conv2d(subspace.shape[0], 64, 5, stride=1, padding="same"),
                                                nn.MaxPool2d(2),
                                                nn.BatchNorm2d(64), nn.Conv2d(64, 64, 3, stride=1, padding="same"),
                                                nn.MaxPool2d(2),
                                                nn.BatchNorm2d(64), 
                                                nn.Conv2d(64, 128, 3, stride=1, padding="same"),
                                                nn.Flatten())
                total_concat_size += 128
            elif key == "other_properties_plane":
                # print(subspace.shape)
                extractors[key] = nn.Sequential(nn.Linear(618, 512), nn.LeakyReLU(), nn.Linear(512, 256))
                # print(key)
                # summary(extractors[key], (1,) + subspace.shape)
                total_concat_size += 128
            elif key == "road_planes":
                extractors[key] = nn.Sequential(nn.Conv2d(subspace.shape[0], 64, 5, stride=1, padding="same"),
                                                nn.MaxPool2d(2),
                                                nn.BatchNorm2d(64), nn.Conv2d(64, 64, 3, stride=1, padding="same"),
                                                nn.BatchNorm2d(64), 
                                                nn.Conv2d(64, 128, 3, stride=1, padding="same"),
                                                nn.MaxPool2d(2),
                                                nn.BatchNorm2d(128), 
                                                nn.Conv2d(128, 256, 3, stride=1, padding="same"),
                                                nn.Flatten())
                total_concat_size += 128
            elif key == "shield_plane":
                extractors[key] = nn.Sequential(nn.Flatten(), nn.Linear(subspace.shape[0] * subspace.shape[1], 512),
                                                nn.LeakyReLU(), nn.Linear(512, 128), nn.LeakyReLU())
                total_concat_size += 64
            elif key == "abbot_planes":
                extractors[key] = nn.Sequential(nn.Conv2d(subspace.shape[0], 64, 5, stride=1, padding="same"),
                                                nn.MaxPool2d(2),
                                                nn.BatchNorm2d(64), 
                                                nn.Conv2d(64, 64, 3, stride=1, padding="same"),
                                                nn.MaxPool2d(2),
                                                nn.BatchNorm2d(64), 
                                                nn.Conv2d(64, 128, 3, stride=1, padding="same"),
                                                nn.Flatten())

                total_concat_size += 64

        total_concat_size = 8768
        self.extractors = nn.ModuleDict(extractors)
        print("Total concat size: ", total_concat_size)
        self._features_dim = total_concat_size

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []

        for key, extractor in self.extractors.items():
            extractor_value = extractor(observations[key])
            encoded_tensor_list.append(extractor_value)
        concated = th.cat(encoded_tensor_list, dim=1)

        return concated


class ExpertDataSet(Dataset):
    def __init__(self, expert_observations, expert_actions):
        self.observations = expert_observations
        self.actions = expert_actions

    def __getitem__(self, index):
        return (self.observations[index], self.actions[index])

    def __len__(self):
        return len(self.observations)


class LazyExpertDataSet(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path

    def __getitem__(self, index):
        with open(self.file_path, "r") as json_file:
            data = json.load(json_file)[index]

            state = data.get("state")
            action = data.get("action")

            if state is not None and action is not None:
                for key, value in state.items():
                    state[key] = np.array(value)

                new_action = []
                new_action.append(one_hot(action[0], 3))
                new_action.append(one_hot(action[1], board_size))
                new_action.append(one_hot(action[2], board_size))
                new_action.append(one_hot(action[3], 4))
                new_action.append(one_hot(action[4], 9))
                new_action.append(one_hot(action[5], 5))

                return state, new_action

    def __len__(self):
        with open(self.file_path, "r") as json_file:
            data = json.load(json_file)
            return len(data)

def numpy_array_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [numpy_array_to_list(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: numpy_array_to_list(value) for key, value in obj.items()}
    else:
        return obj


def one_hot(value, max):
    res = np.zeros(max)
    res[value] = 1

    return torch.tensor(res, dtype=torch.float32).unsqueeze(0).view(-1)

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
    kwargs = {"num_workers": 16, "pin_memory": True} if use_cuda else {}

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
            for sublist in target:
                for tensor in sublist:
                    tensor.to(device)
            # target = target.to(device)
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
                # target = target.long()

            losses = []

            for output, targets in zip(action_prediction, target):
                for o, t in zip(output, targets):
                    # t = torch.tensor(t, dtype=torch.float32).unsqueeze(0).view(-1)
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
    # test_loader = th.utils.data.DataLoader(
    #     dataset=test_expert_dataset,
    #     batch_size=test_batch_size,
    #     shuffle=True,
    #     **kwargs,
    # )

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


if __name__ == '__main__':
    n_players = 2

    board_size = 10

    num_trajectories = 200
    
    env = make_vec_env(CarcassoneEnv, seed=42,  n_envs=16, vec_env_cls=SubprocVecEnv)


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


    policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractor,
        net_arch=[
            dict(pi=[8192], vf=[4096, 1024, 256, 128], dropout=0.5, activation_fn=nn.LeakyReLU)
        ]
    )

    a2c_student = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log="runs/", device="cuda", n_steps=1024,
                      ent_coef=0.02, learning_rate=linear_schedule(0.0003), policy_kwargs=policy_kwargs)

    # a2c_student = PPO.load("model_carcassone")

    for a in range(100):

        for i in range(8):
            env.env_method("set_random", int(time.time()), indices=[i])
            time.sleep(1)

        # trajectories = list(chain.from_iterable(env.env_method("generate_trajectory", 100)))

        trajectories = env.env_method("generate_trajectory", num_trajectories)

        states = []
        actions = []
        for env_trajectories in trajectories:
            for item in env_trajectories:
                state = item.get("state")
                action = item.get("action")

                if state is not None and action is not None:
                    for key, value in state.items():
                        state[key] = np.array(value)
                    states.append(state)

                    new_action = []
                    new_action.append(one_hot(action[0], 3))
                    new_action.append(one_hot(action[1], board_size))
                    new_action.append(one_hot(action[2], board_size))
                    new_action.append(one_hot(action[3], 4))
                    new_action.append(one_hot(action[4], 9))
                    new_action.append(one_hot(action[5], 5))

                    # print(new_action)
                    # new_action =
                    actions.append(new_action)


        expert_dataset = ExpertDataSet(states, actions)  # LazyExpertDataSet("trajectories.json")

        train_size = int(1 * len(expert_dataset))

        test_size = len(expert_dataset) - train_size

        train_expert_dataset, test_expert_dataset = random_split(
            expert_dataset, [train_size, test_size]
        )
        pretrain_agent(
            a2c_student,
            epochs=5,
            scheduler_gamma=0.7,
            learning_rate=1 - (a/(a+2)),
            log_interval=100,
            no_cuda=True,
            seed=1,
            batch_size=512,
            test_batch_size=64,
        )

        mean_reward, _ = evaluate_policy(a2c_student, env, n_eval_episodes=10)

        print(f"Episode: {a + 1}. Mean Reward: {mean_reward}")

        a2c_student.save("pretrained_carcassone3")

   