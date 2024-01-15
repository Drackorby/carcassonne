import json
import numpy as np
from wingedsheep.carcassonne.carcassonne_game import CarcassonneGame  
from wingedsheep.carcassonne.objects.actions.action import Action  
from wingedsheep.carcassonne.tile_sets.supplementary_rules import SupplementaryRule  
from wingedsheep.carcassonne.tile_sets.tile_sets import TileSet  
    
game = CarcassonneGame(  
    players=2,  
    tile_sets=[TileSet.BASE, TileSet.THE_RIVER, TileSet.INNS_AND_CATHEDRALS],  
    supplementary_rules=[SupplementaryRule.ABBOTS, SupplementaryRule.FARMERS],
    visualize=None
)


def numpy_array_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [numpy_array_to_list(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: numpy_array_to_list(value) for key, value in obj.items()}
    else:
        return obj



with open("test_obs.json", "w") as file:

    while not game.is_finished():
        player: int = game.get_current_player()
        valid_actions: [Action] = game.get_possible_actions()
        obs = game.get_obs()
        json.dump(numpy_array_to_list(obs), file)
        break
