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
    def create_action(coded,tile,board,game_state,current_player):
        
        action = None
        action_id = np.argmax(coded[0:3])
        row_id = np.argmax(coded[3:3+len(board)])
        col_id = np.argmax(coded[3+len(board):3+2*len(board)])
        rot_id = np.argmax(coded[3+2*len(board):7+2*len(board)])
        meeple_pos_id = np.argmax(coded[7+2*len(board):16+2*len(board)])
        meeple_type_id = np.argmax(coded[16+2*len(board):21+2*len(board)])
        sides = ["top","right","bottom","left","center","top_left","top_right","bottom_left","bottom_right"]
        meeple_types = ["normal","abbot","farmer","big","big_farmer"]
        coordinate = Coordinate(row_id,col_id)

        if action_id == 0:
            #Tile
            action = TileAction(tile=tile,coordinate=coordinate,tile_rotations=rot_id)
        if action_id == 1:
            #Meeple
            action = MeepleAction(meeple_type=MeepleType(meeple_types[meeple_type_id]),coordinate_with_side=CoordinateWithSide(coordinate,Side(sides[meeple_pos_id])),remove=False)
        if action_id == 2:
            #Remove Meeple or pass
            remove = False

            # Check if it's an Abbot
            if meeple_types[meeple_type_id] == "abbot":
                # Check if an Abbot is placed in the corresponding coordinates
                coordinate_with_side = CoordinateWithSide(coordinate,Side(sides[meeple_pos_id]))
                for placed_meeple in game_state.placed_meeples[current_player]:
                    if placed_meeple.meeple_type == MeepleType.ABBOT and coordinate_with_side== placed_meeple.coordinate_with_side:
                            remove = True

            if remove:
                action = MeepleAction(meeple_type=MeepleType(meeple_types[meeple_type_id]),coordinate_with_side=coordinate_with_side,remove=True)
            else:
                action = PassAction()
        return action

