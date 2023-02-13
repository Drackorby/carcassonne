from stable_baselines3 import A2C,PPO
from stable_baselines3.common.env_util import make_vec_env
from gym_data.envs.Carcassone_env import CarcassoneEnv

from gym.envs.registration import register
import gym 
import torch

# device = torch.device("cuda")
register(id='Carcassone-v0',entry_point='gym_data.envs:CarcassoneEnv',) 

# Parallel environments
env = gym.make('gym_data:Carcassone-v0')

# from stable_baselines3.common.env_checker import check_env
# check_env(env)

# model1 = A2C("MultiInputPolicy", env, verbose=1,device="auto",learning_rate=0.001)
# model2 = A2C("MultiInputPolicy", env, verbose=1,device="auto",learning_rate=0.001)

# model1.set_parameters("carcassone_model12")
# model2.set_parameters("carcassone_model12")

# obs = env.reset()
# done = False

action = [1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0]
from wingedsheep.carcassonne.carcassonne_game import CarcassonneGame  
from wingedsheep.carcassonne.utils.action_util import ActionUtil
from wingedsheep.carcassonne.tile_sets.supplementary_rules import SupplementaryRule
from wingedsheep.carcassonne.tile_sets.tile_sets import TileSet

game = CarcassonneGame(  
            players=2,  board_size = (30,30),
            tile_sets=[TileSet.BASE, TileSet.THE_RIVER, TileSet.INNS_AND_CATHEDRALS],  
            supplementary_rules=[SupplementaryRule.ABBOTS, SupplementaryRule.FARMERS]  
        )     
action = ActionUtil.create_action(action,game.state.next_tile,game.state.board,game.state,0)
possible_actions = game.get_possible_actions()

def gat_types(e):
    return type(e)


for valid_action in possible_actions:
    if type(action) == type(valid_action):
        if action.coordinate == valid_action.coordinate and action.tile_rotations == valid_action.tile_rotations:
            print("valid")

        

# while not done:
#     current_player = env.get_current_player()
#     print(current_player)
#     if current_player == 0:
#         action, _state = model1.predict(obs, deterministic=False)

#     else:
#         action, _state = model1.predict(obs, deterministic=False)

#     print(action)
#     obs, reward, done, info = env.test_step(action,current_player)
#     env.render(mode="human")