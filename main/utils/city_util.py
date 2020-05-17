from typing import Set

from main.carcassonne_game_state import CarcassonneGameState
from main.objects.city import City
from main.objects.coordinate import Coordinate
from main.objects.coordinate_with_side import CoordinateWithSide
from main.objects.meeple_position import MeeplePosition
from main.objects.side import Side
from main.objects.terrain_type import TerrainType
from main.objects.tile import Tile


class CityUtil:

    def __init__(self):
        pass

    def find_city(self, game_state: CarcassonneGameState, city_position: CoordinateWithSide) -> City:
        cities: Set[CoordinateWithSide] = set(self.cities_for_position(game_state, city_position))
        open_edges: Set[CoordinateWithSide] = set(map(lambda x: self.opposite_edge(x), cities))
        explored: Set[CoordinateWithSide] = cities.union(open_edges)
        while len(open_edges) > 0:
            open_edge: CoordinateWithSide = open_edges.pop()
            new_cities = self.cities_for_position(game_state, open_edge)
            cities = cities.union(new_cities)
            new_open_edges = set(map(lambda x: self.opposite_edge(x), new_cities))
            explored = explored.union(new_cities)
            new_open_edge: CoordinateWithSide
            for new_open_edge in new_open_edges:
                if new_open_edge not in explored:
                    open_edges.add(new_open_edge)
                    explored.add(new_open_edge)

        finished: bool = len(explored) == len(cities)
        return City(city_positions=cities, finished=finished)

    def opposite_edge(self, city_position: CoordinateWithSide):
        if city_position.side == Side.TOP:
            return CoordinateWithSide(Coordinate(city_position.coordinate.row - 1, city_position.coordinate.column),
                                      Side.BOTTOM)
        elif city_position.side == Side.RIGHT:
            return CoordinateWithSide(Coordinate(city_position.coordinate.row, city_position.coordinate.column + 1),
                                      Side.LEFT)
        elif city_position.side == Side.BOTTOM:
            return CoordinateWithSide(Coordinate(city_position.coordinate.row + 1, city_position.coordinate.column),
                                      Side.TOP)
        elif city_position.side == Side.LEFT:
            return CoordinateWithSide(Coordinate(city_position.coordinate.row, city_position.coordinate.column - 1),
                                      Side.RIGHT)

    def cities_for_position(self, game_state: CarcassonneGameState, city_position: CoordinateWithSide):
        tile: Tile = game_state.board[city_position.coordinate.row][city_position.coordinate.column]
        cities = []
        if tile is None:
            return cities
        for city_group in tile.city:
            if city_position.side in city_group:
                city_group_side: Side
                for city_group_side in city_group:
                    city_position: CoordinateWithSide = CoordinateWithSide(city_position.coordinate, city_group_side)
                    cities.append(city_position)
        return cities

    def city_contains_meeples(self, game_state: CarcassonneGameState, city: City):
        for city_position in city.city_positions:
            for i in range(game_state.players):
                if city_position in list(map(lambda x: x.coordinate_with_side, game_state.placed_meeples[i])):
                    return True
        return False

    def find_meeples(self, game_state: CarcassonneGameState, city: City) -> [[MeeplePosition]]:
        meeples: [[MeeplePosition]] = []

        for i in range(game_state.players):
            meeples.append([])

        for city_position in city.city_positions:
            for i in range(game_state.players):
                meeple_position: MeeplePosition
                for meeple_position in game_state.placed_meeples[i]:
                    if city_position == meeple_position.coordinate_with_side:
                        meeples[i].append(meeple_position)

        return meeples

    def find_cities(self, game_state: CarcassonneGameState, coordinate: Coordinate):
        cities: [City] = []

        tile: Tile = game_state.board[coordinate.row][coordinate.column]

        if tile is None:
            return cities

        side: Side
        for side in [Side.TOP, Side.RIGHT, Side.BOTTOM, Side.LEFT]:
            if tile.get_type(side) == TerrainType.CITY:
                city: City = self.find_city(game_state=game_state,
                                            city_position=CoordinateWithSide(coordinate=coordinate, side=side))
                cities.append(city)

        return cities