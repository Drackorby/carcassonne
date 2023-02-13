import numpy as np

import gym
from gym import spaces

import random

from wingedsheep.carcassonne.carcassonne_game_state import CarcassonneGameState
from wingedsheep.carcassonne.carcassonne_visualiser import CarcassonneVisualiser
from wingedsheep.carcassonne.objects.actions.action import Action
from wingedsheep.carcassonne.tile_sets.supplementary_rules import SupplementaryRule
from wingedsheep.carcassonne.tile_sets.tile_sets import TileSet
from wingedsheep.carcassonne.utils.action_util import ActionUtil
from wingedsheep.carcassonne.utils.state_updater import StateUpdater
from wingedsheep.carcassonne.carcassonne_game import CarcassonneGame  
from wingedsheep.carcassonne.utils.points_collector import PointsCollector

    
class CarcassoneEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array",None], "render_fps": 4}

    def __init__(self, render_mode=None):

        n_players = 2

        board_size = 30


        self.game = CarcassonneGame(  
            players=n_players,  board_size = (board_size,board_size),
            tile_sets=[TileSet.BASE, TileSet.THE_RIVER, TileSet.INNS_AND_CATHEDRALS],  
            supplementary_rules=[SupplementaryRule.ABBOTS, SupplementaryRule.FARMERS]  
        )         
        # self.game.state.get_obs()

        other_properties_space = np.ones(16)*2

        other_properties_space[-3] = 250
        other_properties_space[-2] = n_players
        other_properties_space[-1] = 3
        other_properties_space[0] = 15
        other_properties_space[1] = 10
        other_properties_space[5] = 59
        other_properties_space[6] = 8 # Meeples
        other_properties_space[7] = 8

        self.observation_space = spaces.Dict(
        {
            "city_planes": spaces.MultiBinary(15*board_size*board_size),
            "road_planes": spaces.MultiBinary(10*board_size*board_size),
            "chapel_plane": spaces.MultiBinary(board_size*board_size),
            "shield_plane": spaces.MultiBinary(board_size*board_size),
            "flowers_plane": spaces.MultiBinary(board_size*board_size),
            "field_planes": spaces.MultiBinary(59*board_size*board_size),
            "meeple_planes": spaces.MultiBinary(5*n_players*board_size*board_size),
            "abbot_planes": spaces.MultiBinary(n_players*board_size*board_size),
            "farmer_planes": spaces.MultiBinary(9*n_players*board_size*board_size),
            "big_farmer_planes": spaces.MultiBinary(9*n_players*board_size*board_size),
            "big_meeples_planes": spaces.MultiBinary(5*n_players*board_size*board_size),
            "other_properties_plane": spaces.MultiDiscrete(other_properties_space)}
        )
        

        self.action_space = spaces.MultiBinary(3+ 2*board_size+ 4 + 9 +5)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None



    def reset(self):
        self.game.reset()
        observation = self.game.get_obs()
        return observation

    def test_step(self,action,player_id):

        action = ActionUtil.create_action(action,self.game.state.next_tile,self.game.state.board,self.game.state,player_id)
        valid_actions = self.game.get_possible_actions() 

        reward = -1000

        if action in valid_actions:
            self.game.step(player_id,action)
        
            scores = PointsCollector.count_score(self.game.state)
            my_score = scores[player_id]
            del scores[player_id]
            reward = my_score - np.max(scores)


        observation = self.game.get_obs()
        terminated = self.game.is_finished()
        info = {}

        return observation, reward, terminated, info
    

    def get_current_player(self):
        return self.game.get_current_player()  
    def step(self,action,player_id = 0):

        action = ActionUtil.create_action(action,self.game.state.next_tile,self.game.state.board,self.game.state,player_id)
        valid_actions = self.game.get_possible_actions() 

        reward = -100

        if action in valid_actions:
            self.game.step(player_id,action)
        
            scores = PointsCollector.count_score(self.game.state)
            my_score = scores[player_id]
            del scores[player_id]
            reward = my_score - np.max(scores)

        observation = self.game.get_obs()
        terminated = self.game.is_finished()
        info = {}

        player = self.game.get_current_player()  
        while player != 0:
            valid_actions = self.game.get_possible_actions()  
            action = random.choice(valid_actions)  
            if action is not None:  
                self.game.step(player, action) 

        return observation, reward, terminated, info

    def render(self,mode,**kwargs):
        # pass
        # if self.render_mode == "human":
        self.game.render()

    def close(self):
        pass



