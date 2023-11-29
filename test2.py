from stable_baselines3 import A2C

from gym.envs.registration import register
import gym 

from stable_baselines3 import A2C
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from gym.envs.registration import register

from gym import spaces
import torch as th
from torch import nn
from sb3_contrib.ppo_mask import MaskablePPO

# device = torch.device("cuda")
register(id='Carcassone-v0',entry_point='gym_data.envs:CarcassoneEnv',) 

from gym_data.envs import CarcassoneEnv
import numpy  as np
from sb3_contrib.common.wrappers import ActionMasker


def mask_fn(env: gym.Env) -> np.ndarray:
    # Do whatever you'd like in this function to return the action mask
    # for the current env. In this example, we assume the env has a
    # helpful method we can rely on.
    return env.valid_action_mask()


def make_env():
    env = CarcassoneEnv()
    env = ActionMasker(env, mask_fn)
    return env


# Parallel environments
env = gym.make('gym_data:Carcassone-v0')

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
            if key == "image":
                # We will just downsample one channel of the image by 4x4 and flatten.
                # Assume the image is single-channel (subspace.shape[0] == 0)
                extractors[key] = nn.Sequential(nn.MaxPool2d(4), nn.Flatten())
                total_concat_size += subspace.shape[1] // 4 * subspace.shape[2] // 4
            elif key == "vector":
                # Run through a simple MLP
                extractors[key] = nn.Linear(subspace.shape[0], 128)
                total_concat_size += 128
            elif key == "big_farmer_planes":
                extractors[key] = nn.Sequential(nn.Linear(subspace.shape[0], 2048),nn.Linear(2048, 512), nn.Linear(512, 128)) 
                total_concat_size += 128
            elif key == "big_meeples_planes":
                extractors[key] = nn.Sequential(nn.Linear(subspace.shape[0], 512), nn.Linear(512, 128)) 
                total_concat_size += 128
            elif key == "chapel_plane":
                extractors[key] = nn.Linear(subspace.shape[0], 128)
                total_concat_size += 128
            elif key == "city_planes":
                extractors[key] = nn.Sequential(nn.Linear(subspace.shape[0], 1024),nn.Linear(1024, 512), nn.Linear(512, 128)) 
                total_concat_size += 128
            elif key == "farmer_planes":
                extractors[key] = nn.Sequential(nn.Linear(subspace.shape[0], 1024),nn.Linear(1024, 512), nn.Linear(512, 128)) 
                total_concat_size += 128
            elif key == "field_planes":
                extractors[key] = nn.Sequential(nn.Linear(subspace.shape[0], 1024),nn.Linear(1024, 512), nn.Linear(512, 128)) 
                total_concat_size += 128
            elif key == "flowers_plane":
                extractors[key] = nn.Linear(subspace.shape[0], 128)
                total_concat_size += 128
            elif key == "meeple_planes":
                extractors[key] = nn.Sequential(nn.Linear(subspace.shape[0], 512), nn.Linear(512, 128)) 
                total_concat_size += 128
            elif key == "other_properties_plane":
                extractors[key] = nn.Linear(371, 128)
                total_concat_size += 128
            elif key == "road_planes":
                extractors[key] = nn.Sequential(nn.Linear(subspace.shape[0], 512), nn.Linear(512, 128)) 
            elif key == "shield_plane":
                extractors[key] = nn.Linear(subspace.shape[0], 128)
                total_concat_size += 128
            elif key == "abbot_planes":
                extractors[key] = nn.Linear(subspace.shape[0], 128)
                total_concat_size += 128


        self.extractors = nn.ModuleDict(extractors)
        # print("Total concat size: " , total_concat_size)
        # Update the features dim manually
        self._features_dim = total_concat_size

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


policy_kwargs = dict(
    features_extractor_class=CustomCombinedExtractor,
    net_arch = [2048, dict(pi=[ 256, 128], vf=[512, 64])]
)


model1 = MaskablePPO("MultiInputPolicy", env,policy_kwargs=policy_kwargs, verbose=1,device="auto", n_steps=256, batch_size=64, n_epochs=1) # , learning_rate=0.002)
model2 = MaskablePPO("MultiInputPolicy", env,policy_kwargs=policy_kwargs, verbose=1,device="auto", n_steps=256, batch_size=64, n_epochs=1) # , learning_rate=0.002)


model1.set_parameters("carcassone_model_mask_PPO")
model2.set_parameters("carcassone_model_mask_PPO")

obs = env.reset()
done = False


while not done:
    current_player = env.get_current_player()
    print(current_player)
    if current_player == 0:
        action, _state = model1.predict(obs, deterministic=False)


    else:
        action, _state = model2.predict(obs, deterministic=False)

    obs, reward, done, info = env.test_step(action,current_player)
    env.render(mode="human")







# action = [1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0]

# from wingedsheep.carcassonne.objects.actions.tile_action import TileAction
# print(type(TileAction))

# game = CarcassonneGame(  
#             players=2,  board_size = (30,30),
#             tile_sets=[TileSet.BASE, TileSet.THE_RIVER, TileSet.INNS_AND_CATHEDRALS],  
#             supplementary_rules=[SupplementaryRule.ABBOTS, SupplementaryRule.FARMERS]  
#         )     
# action = ActionUtil.create_action(action,game.state.next_tile,game.state.board,game.state,0)
# possible_actions = game.get_possible_actions()

# def gat_types(e):
#     return type(e)


# for valid_action in possible_actions:
#     if type(action) == type(valid_action):
#         if action.coordinate == valid_action.coordinate and action.tile_rotations == valid_action.tile_rotations:
#             print("valid")

        

