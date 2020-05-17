import unittest

from main.carcassonne_game_state import CarcassonneGameState
from main.decks.base_deck import base_tiles
from main.objects.city import City
from main.objects.coordinate import Coordinate
from main.objects.coordinate_with_side import CoordinateWithSide
from main.objects.meeple_position import MeeplePosition
from main.objects.meeple_type import MeepleType
from main.objects.side import Side
from main.utils.city_util import CityUtil


class TestCityUtil(unittest.TestCase):

    def test_find_city(self):
        """
        Find the positions for a city with a top and a bottom
        """

        # Given
        city_util: CityUtil = CityUtil()
        game_state: CarcassonneGameState = CarcassonneGameState()

        city_top = base_tiles["city_one_side"]
        city_bottom = city_top.turn(2)

        game_state.board = []
        for row in range(2):
            game_state.board.append([])
            for column in range(1):
                game_state.board[row].append(None)

        game_state.board[0][0] = city_bottom
        game_state.board[1][0] = city_top

        # When
        city: City = city_util.find_city(
            game_state=game_state,
            city_position=CoordinateWithSide(Coordinate(0, 0), Side.BOTTOM)
        )

        # Then
        self.assertTrue(city.finished)
        self.assertEqual(2, len(city.city_positions))
        self.assertIn(CoordinateWithSide(Coordinate(0, 0), Side.BOTTOM), city.city_positions)
        self.assertIn(CoordinateWithSide(Coordinate(1, 0), Side.TOP), city.city_positions)


    def test_find_donut_city(self):
        """
        Find the positions for a donut shaped city
        """

        # Given
        city_util: CityUtil = CityUtil()
        game_state: CarcassonneGameState = self.create_donut_city_board()

        # When
        city: City = city_util.find_city(
            game_state=game_state,
            city_position=CoordinateWithSide(Coordinate(0, 0), Side.BOTTOM)
        )

        # Then
        self.assertTrue(city.finished)
        self.assertEqual(16, len(city.city_positions))
        self.assertIn(CoordinateWithSide(Coordinate(0, 0), Side.BOTTOM), city.city_positions)
        self.assertIn(CoordinateWithSide(Coordinate(0, 0), Side.RIGHT), city.city_positions)
        self.assertIn(CoordinateWithSide(Coordinate(0, 1), Side.LEFT), city.city_positions)
        self.assertIn(CoordinateWithSide(Coordinate(0, 1), Side.RIGHT), city.city_positions)
        self.assertIn(CoordinateWithSide(Coordinate(0, 2), Side.LEFT), city.city_positions)
        self.assertIn(CoordinateWithSide(Coordinate(0, 2), Side.BOTTOM), city.city_positions)
        self.assertIn(CoordinateWithSide(Coordinate(1, 0), Side.TOP), city.city_positions)
        self.assertIn(CoordinateWithSide(Coordinate(1, 0), Side.BOTTOM), city.city_positions)
        self.assertIn(CoordinateWithSide(Coordinate(1, 2), Side.TOP), city.city_positions)
        self.assertIn(CoordinateWithSide(Coordinate(1, 2), Side.BOTTOM), city.city_positions)
        self.assertIn(CoordinateWithSide(Coordinate(2, 0), Side.TOP), city.city_positions)
        self.assertIn(CoordinateWithSide(Coordinate(2, 0), Side.RIGHT), city.city_positions)
        self.assertIn(CoordinateWithSide(Coordinate(2, 1), Side.LEFT), city.city_positions)
        self.assertIn(CoordinateWithSide(Coordinate(2, 1), Side.RIGHT), city.city_positions)
        self.assertIn(CoordinateWithSide(Coordinate(2, 2), Side.LEFT), city.city_positions)
        self.assertIn(CoordinateWithSide(Coordinate(2, 2), Side.TOP), city.city_positions)

    def test_find_meeples_in_donut_city(self):
        """
        Find meeple positions for a donut shaped city
        """

        # Given
        city_util: CityUtil = CityUtil()
        game_state: CarcassonneGameState = self.create_donut_city_board()
        game_state.placed_meeples = [[], [], []]
        game_state.placed_meeples[0].append(MeeplePosition(meeple_type=MeepleType.NORMAL, coordinate_with_side=CoordinateWithSide(Coordinate(2, 1), Side.RIGHT)))
        game_state.placed_meeples[0].append(MeeplePosition(meeple_type=MeepleType.NORMAL, coordinate_with_side=CoordinateWithSide(Coordinate(0, 1), Side.LEFT)))
        game_state.placed_meeples[1].append(MeeplePosition(meeple_type=MeepleType.BIG, coordinate_with_side=CoordinateWithSide(Coordinate(1, 2), Side.TOP)))
        game_state.players = 3

        # When
        city: City = city_util.find_city(
            game_state=game_state,
            city_position=CoordinateWithSide(Coordinate(0, 0), Side.BOTTOM)
        )

        meeples: [[MeeplePosition]] = city_util.find_meeples(game_state, city)

        # Then
        self.assertEqual(3, len(meeples))
        self.assertEqual(2, len(meeples[0]))
        self.assertEqual(1, len(meeples[1]))
        self.assertEqual(0, len(meeples[2]))
        self.assertIn(MeeplePosition(MeepleType.NORMAL, CoordinateWithSide(Coordinate(2, 1), Side.RIGHT)), meeples[0])
        self.assertIn(MeeplePosition(MeepleType.NORMAL, CoordinateWithSide(Coordinate(0, 1), Side.LEFT)), meeples[0])
        self.assertIn(MeeplePosition(MeepleType.BIG, CoordinateWithSide(Coordinate(1, 2), Side.TOP)), meeples[1])

    def create_donut_city_board(self) -> CarcassonneGameState:
        game_state = CarcassonneGameState()
        city_narrow_left_right = base_tiles["city_narrow"]
        city_narrow_top_bottom = base_tiles["city_narrow"].turn(1)
        city_diagonal_top_right = base_tiles["city_diagonal_top_right"]
        city_diagonal_bottom_right = base_tiles["city_diagonal_top_right"].turn(1)
        city_diagonal_bottom_left = base_tiles["city_diagonal_top_right"].turn(2)
        city_diagonal_top_left = base_tiles["city_diagonal_top_right"].turn(3)
        game_state.board = []
        for row in range(3):
            game_state.board.append([])
            for column in range(3):
                game_state.board[row].append(None)
        game_state.board[0][0] = city_diagonal_bottom_right
        game_state.board[0][1] = city_narrow_left_right
        game_state.board[0][2] = city_diagonal_bottom_left
        game_state.board[1][0] = city_narrow_top_bottom
        game_state.board[1][2] = city_narrow_top_bottom
        game_state.board[2][0] = city_diagonal_top_right
        game_state.board[2][1] = city_narrow_left_right
        game_state.board[2][2] = city_diagonal_top_left
        return game_state