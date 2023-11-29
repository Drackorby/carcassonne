from wingedsheep.carcassonne.carcassonne_game_state import CarcassonneGameState
from wingedsheep.carcassonne.carcassonne_visualiser import CarcassonneVisualiser
from wingedsheep.carcassonne.objects.actions.action import Action
from wingedsheep.carcassonne.tile_sets.supplementary_rules import SupplementaryRule
from wingedsheep.carcassonne.tile_sets.tile_sets import TileSet
from wingedsheep.carcassonne.utils.action_util import ActionUtil
from wingedsheep.carcassonne.utils.state_updater import StateUpdater

import numpy as np
import random
import copy

class CarcassonneGame:

    def __init__(self,
                 players: int = 2, board_size: (int, int) = (10, 10),
                 tile_sets: [TileSet] = (TileSet.BASE, TileSet.THE_RIVER, TileSet.INNS_AND_CATHEDRALS),
                 supplementary_rules: [SupplementaryRule] = (SupplementaryRule.FARMERS, SupplementaryRule.ABBOTS), visualize = True):
        self.random_seed = 0
        
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        
        self.players = players
        self.tile_sets = tile_sets
        self.supplementary_rules = supplementary_rules
        self.state: CarcassonneGameState = CarcassonneGameState(
            tile_sets=tile_sets,
            players=players,
            supplementary_rules=supplementary_rules,
            board_size=board_size
        )
        self.visualize = visualize
        
        self.visualiser = CarcassonneVisualiser() if visualize else None

    def reset(self,rand_seed = -1):
        random.seed(self.random_seed if rand_seed != -1 else self.random_seed)
        np.random.seed(self.random_seed if rand_seed != -1 else self.random_seed)
        self.state = CarcassonneGameState(tile_sets=self.tile_sets, supplementary_rules=self.supplementary_rules)

    def step(self, player: int, action: Action):
        self.state = StateUpdater.apply_action(game_state=self.state, action=action)

    # def step_copy(self, action: Action):
    #     game = copy.copy(self)
    #     game.state = StateUpdater.apply_action(game_state=game.state, action=action)
    #     return game

    def render(self):
        if self.visualize:
            self.visualiser.draw_game_state(self.state)

    def is_finished(self) -> bool:
        return self.state.is_terminated()

    def get_current_player(self) -> int:
        return self.state.current_player
    
    def get_obs(self, reverse_player=False):
        return self.state.get_obs(reverse_player)

    def get_possible_actions(self) -> [Action]:
        return ActionUtil.get_possible_actions(self.state)

    def get_reward(self, player):
        return self.state.scores[player]
    
    def update_random_num(self, rand_num):
        self.random_seed = rand_num
        random.seed(rand_num)
        np.random.seed(rand_num)
