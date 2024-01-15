
# This file conteins function that allow the checking of observations and actions to see if they are currently being coded and decoded correctly

from wingedsheep.carcassonne.carcassonne_game import CarcassonneGame
from wingedsheep.carcassonne.utils.action_util import ActionUtil
from wingedsheep.carcassonne.tile_sets.supplementary_rules import SupplementaryRule
from wingedsheep.carcassonne.objects.actions.tile_action import TileAction
from wingedsheep.carcassonne.tile_sets.tile_sets import TileSet
import random
import numpy as np


def validate_data(data, shapes):
    other_properties_space = np.ones(16) * 2

    other_properties_space[-3] = 250
    other_properties_space[-4] = 250
    other_properties_space[-2] = 2
    other_properties_space[-1] = 3
    other_properties_space[0] = 15
    other_properties_space[1] = 10
    other_properties_space[5] = 59
    other_properties_space[6] = 8  # Meeples
    other_properties_space[7] = 8


    warnings = []
    for key in data:
        if key not in shapes:
            warnings.append(f"Key '{key}' does not exist in shapes dictionary.")
        else:
            if key == "other_properties_plane":
                for i, e in enumerate(data[key]):
                    if other_properties_space[i] < data[key][i]:
                        warnings.append(
                            f"Value for key '{key}' has incorrect shape. At: {i} with expected < {other_properties_space[i]}, Actual: {data[key][i]}")
            expected_shape = shapes[key]
            data_shape = np.shape(data[key])
            if data_shape != expected_shape:
                warnings.append(f"Value for key '{key}' has incorrect shape. Expected: {expected_shape}, Actual: {data_shape}")
    return warnings



def check_observation():
    # Test coding and decoding observations

    n_players = 2
    board_size = 30

    other_properties_space_shape = (16,)

    observation_space_shape = {
            "city_planes": (15, board_size, board_size),
            "road_planes": (10, board_size, board_size),
            "chapel_plane": (board_size, board_size),
            "shield_plane": (board_size, board_size),
            "flowers_plane": (board_size, board_size),
            "field_planes": (59, board_size, board_size),
            "meeple_planes": (5 * n_players, board_size, board_size),
            "abbot_planes": (n_players, board_size, board_size),
            "farmer_planes": (9 * n_players, board_size, board_size),
            "big_farmer_planes": (9 * n_players, board_size, board_size),
            "big_meeples_planes": (5 * n_players, board_size, board_size),
            "other_properties_plane": (other_properties_space_shape)}

    game = CarcassonneGame(
        players=n_players, board_size=(board_size, board_size),
        tile_sets=[TileSet.BASE, TileSet.THE_RIVER, TileSet.INNS_AND_CATHEDRALS],
        supplementary_rules=[SupplementaryRule.ABBOTS, SupplementaryRule.FARMERS], visualize=False
    )
    while not game.is_finished():
        possible_actions = game.get_possible_actions()
        player = game.get_current_player()
        obs = game.state.get_obs()

        warnings = validate_data(obs, observation_space_shape)
        for warning in warnings:
            print(warning)

        action = random.choice(possible_actions)
        if action is not None:
            game.step(player, action)


def check_action():
    # Test coding and decoding actions
    game = CarcassonneGame(
        players=2, board_size=(30, 30),
        tile_sets=[TileSet.BASE, TileSet.THE_RIVER, TileSet.INNS_AND_CATHEDRALS],
        supplementary_rules=[SupplementaryRule.ABBOTS, SupplementaryRule.FARMERS], visualize=False
    )
    while not game.is_finished():
        possible_actions = game.get_possible_actions()
        player = game.get_current_player()

        for a in possible_actions:
            coded_action = ActionUtil.code_action(a, game.state.board)
            decoded_action = ActionUtil.create_action(coded_action, a.tile if isinstance(a, TileAction) else None,
                                                      game.state.board, game.state, player)
            diff = abs(np.linalg.norm(ActionUtil.code_action(decoded_action, game.state.board) - coded_action))
            if diff != 0 or decoded_action != a:
                print(diff)
                print("Dedoded action different than original")
                print("Coded action", coded_action)
                print("Original action", a, a.print_attributes())
                print("Decoded action", decoded_action, decoded_action.print_attributes())
                ActionUtil.create_action(coded_action, a.tile if isinstance(a, TileAction) else None, game.state.board,
                                         game.state, player, True)

        action = random.choice(possible_actions)
        if action is not None:
            game.step(player, action)
