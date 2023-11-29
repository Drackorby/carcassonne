from wingedsheep.carcassonne.carcassonne_game_state import CarcassonneGameState, GamePhase
from wingedsheep.carcassonne.objects.actions.action import Action
from wingedsheep.carcassonne.objects.actions.pass_action import PassAction
from wingedsheep.carcassonne.objects.actions.tile_action import TileAction
from wingedsheep.carcassonne.objects.actions.meeple_action import MeepleAction
from wingedsheep.carcassonne.objects.playing_position import PlayingPosition
from wingedsheep.carcassonne.utils.possible_move_finder import PossibleMoveFinder
from wingedsheep.carcassonne.utils.tile_position_finder import TilePositionFinder
import numpy as np
from wingedsheep.carcassonne.objects.coordinate import Coordinate
from wingedsheep.carcassonne.objects.meeple_type import MeepleType
from wingedsheep.carcassonne.objects.coordinate_with_side import CoordinateWithSide
from wingedsheep.carcassonne.objects.side import Side
import random


class ActionUtil:

    @staticmethod
    def get_possible_actions(state: CarcassonneGameState):
        actions: [Action] = []
        if state.phase == GamePhase.TILES:
            possible_playing_positions: [PlayingPosition] = TilePositionFinder.possible_playing_positions(
                game_state=state,
                tile_to_play=state.next_tile
            )
            if len(possible_playing_positions) == 0:
                actions.append(PassAction())
            else:
                playing_position: PlayingPosition
                for playing_position in possible_playing_positions:
                    action = TileAction(
                        tile=state.next_tile.turn(playing_position.turns),
                        coordinate=playing_position.coordinate,
                        tile_rotations=playing_position.turns
                    )
                    actions.append(action)
        elif state.phase == GamePhase.MEEPLES:
            possible_meeple_actions = PossibleMoveFinder.possible_meeple_actions(game_state=state)
            actions.extend(possible_meeple_actions)
            actions.append(PassAction())
        return actions

    @staticmethod
    def generate_random_action():
        mask = np.zeros(6, dtype=int)
        mask[0] = int(random.randint(0, 2))
        row = int(random.randint(0, 9))
        column = int(random.randint(0, 9))
        rot_id = int(random.randint(0, 3))
        mask[1] = int(row)
        mask[2] = int(column)
        mask[3] = int(rot_id)
        mask[4] = int(random.randint(0, 8))
        mask[5] = int(random.randint(0, 4))

        return mask

    @staticmethod
    def code_action(action):
        mask = np.zeros(6, dtype=int)
        if isinstance(action, TileAction):
            mask[0] = int(0)
            row = int(action.coordinate.row)
            column = int(action.coordinate.column)
            rot_id = int(action.tile_rotations)
            mask[1] = int(row)
            mask[2] = int(column)
            mask[3] = int(rot_id)

        elif isinstance(action, MeepleAction):
            mask[0] = int(1)
            sides = ["top", "right", "bottom", "left", "center", "top_left", "top_right", "bottom_left", "bottom_right"]
            meeple_types = ["normal", "abbot", "farmer", "big", "big_farmer"]

            row = action.coordinate_with_side.coordinate.row
            column = action.coordinate_with_side.coordinate.column
            side = action.coordinate_with_side.side

            side = sides.index(side.__str__())
            meeple_type = meeple_types.index(action.meeple_type.__str__())

            mask[1] = int(row)
            mask[2] = int(column)
            mask[4] = int(side)
            mask[5] = int(meeple_type)

        elif isinstance(action, PassAction):
            mask[0] = int(2)
        else:
            print("error in mask: ", action)

        return mask

    @staticmethod
    def get_all_possible_coded_actions(actions, board_size):
        sides = ["top", "right", "bottom", "left", "center", "top_left", "top_right", "bottom_left", "bottom_right"]
        meeple_types = ["normal", "abbot", "farmer", "big", "big_farmer"]

        all_actions = []

        for action in actions:
            if action[0] == 0:  # Tile action
                for i in range(len(sides)):
                    for j in range(len(meeple_types)):
                        new_action = action.copy()
                        new_action[4] = i
                        new_action[5] = j
                        all_actions.append(new_action)
            elif action[0] == 1:  # Meeple action
                for i in range(4):
                    new_action = action.copy()
                    new_action[3] = i
                    all_actions.append(new_action)
            elif action[0] == 2:  # Pass action
                range1 = np.arange(board_size)
                range2 = np.arange(board_size)
                range3 = np.arange(4)
                range4 = np.arange(len(sides))
                range5 = np.arange(len(meeple_types))

                for i in range1:
                    for j in range2:
                        for k in range3:
                            for l in range4:
                                for m in range5:
                                    new_action = action.copy()
                                    new_action[1] = i
                                    new_action[2] = j
                                    new_action[3] = k
                                    new_action[4] = l
                                    new_action[5] = m
                                    all_actions.append(new_action)

        return all_actions
    # def get_all_possible_coded_actions(actions, board_size):
    #     # If transforming the actions into a single value we need to be able to infer that some part is independent
    #     # given the type of action
    #     all_actions = np.array()
    #
    #     sides = ["top", "right", "bottom", "left", "center", "top_left", "top_right", "bottom_left", "bottom_right"]
    #     meeple_types = ["normal", "abbot", "farmer", "big", "big_farmer"]
    #
    #     for action in actions:
    #         if action[0] == 0:  # Tile action
    #             temp = []
    #             for i in range(len(sides)):
    #                 for j in range(len(meeple_types)):
    #                     new_action = np.copy(action)
    #
    #                     new_action[4] = i
    #                     new_action[5] = j
    #
    #                     temp.append(new_action)
    #             all_actions = np.concatenate((all_actions, np.array(temp)))
    #         if action[0] == 1:  # Meeple action
    #             temp = []
    #             for i in range(4):
    #                 new_action = np.copy(action)
    #
    #                 new_action[3] = i
    #
    #                 temp.append(new_action)
    #             all_actions = np.concatenate((all_actions, np.array(temp)))
    #
    #
    #         elif action[0] == 2:  # Pass action
    #             range1 = np.arange(board_size)
    #             range2 = np.arange(board_size)
    #             range3 = np.arange(4)
    #             range4 = np.arange(len(sides))
    #             range5 = np.arange(len(meeple_types))
    #
    #             # Create a first array filled with a specific value, such as 0
    #             first_array_value = 2
    #             first_array = np.full((len(range1), len(range2), len(range3), len(range4), len(range5)),
    #                                   first_array_value, dtype=int)
    #
    #             # Use meshgrid to create all combinations for the remaining arrays
    #             arr1, arr2, arr3, arr4, arr5 = np.meshgrid(range1, range2, range3, range4, range5)
    #
    #             # Stack the arrays, including the first array, to create the final array
    #             result = np.stack((first_array, arr1, arr2, arr3,arr4,arr5), axis=-1)
    #             all_actions = np.concatenate((all_actions, result))
    #
    #     return all_actions

    @staticmethod
    def create_action(coded, tile, board, game_state, current_player, verbose=False):

        action = None
        action_id = int(coded[0])
        row_id = int(coded[1])
        col_id = int(coded[2])
        rot_id = int(coded[3])
        meeple_pos_id = int(coded[4])
        meeple_type_id = int(coded[5])
        sides = ["top", "right", "bottom", "left", "center", "top_left", "top_right", "bottom_left", "bottom_right"]
        meeple_types = ["normal", "abbot", "farmer", "big", "big_farmer"]
        coordinate = Coordinate(row_id, col_id)
        # if np.random.random() > 0.8:
        #     print(action_id, row_id, col_id, rot_id, meeple_pos_id, meeple_type_id)

        if action_id == 0:
            # Tile
            action = TileAction(tile=tile, coordinate=coordinate, tile_rotations=rot_id)
            
        if action_id == 1:
            # Meeple action
            remove = False
            if verbose:
                print("meeple type (create function): ", meeple_types[meeple_type_id])

            # Check if it's an Abbot
            if meeple_types[meeple_type_id] == "abbot":
                # Check if an Abbot is placed in the corresponding coordinates
                coordinate_with_side = CoordinateWithSide(coordinate, Side(sides[meeple_pos_id]))
                for placed_meeple in game_state.placed_meeples[current_player]:
                    if placed_meeple.meeple_type == MeepleType.ABBOT and coordinate_with_side == placed_meeple.coordinate_with_side:
                        remove = True

            if remove:
                action = MeepleAction(meeple_type=MeepleType(meeple_types[meeple_type_id]),
                                      coordinate_with_side=coordinate_with_side, remove=True)
            else:
                action = MeepleAction(meeple_type=MeepleType(meeple_types[meeple_type_id]),
                                      coordinate_with_side=CoordinateWithSide(coordinate, Side(sides[meeple_pos_id])),
                                      remove=False)

        if action_id == 2:
            # Pass
            action = PassAction()

        return action
