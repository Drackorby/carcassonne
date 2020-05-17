import json
import sys
from typing import Set
import numpy as np

from main.objects.connection import Connection
from main.objects.side import Side
from main.objects.terrain_type import TerrainType
from main.utils.connection_modification import turn_connection
from main.utils.side_modification import turn_side, turn_sides

np.set_printoptions(suppress=True, linewidth=np.nan, threshold=sys.maxsize)


class Tile:
    def __init__(self,
                 description: str = "",
                 turns: int = 0,
                 road: [Connection] = (),
                 river: [Connection] = (),
                 city: [[Side]] = (),
                 grass: [Side] = (),
                 shield: bool = False,
                 chapel_or_flowers: bool = False,
                 inn: [Side] = (),
                 cathedral: bool = False,
                 unplayable_sides: [Side] = (),
                 image: str = "Empty.png"):
        self.description = description
        self.turns = turns
        self.road = road
        self.river = river
        self.city = city
        self.grass = grass
        self.shield = shield
        self.chapel_or_flowers = chapel_or_flowers
        self.inn = inn
        self.cathedral = cathedral
        self.unplayable_sides = unplayable_sides
        self.image = image

    def get_road_ends(self) -> Set[Side]:
        sides: Set[Side] = set([])
        for road in self.road:
            sides.add(road.a)
            sides.add(road.b)
        return set(sides)

    def get_river_ends(self) -> Set[Side]:
        sides: Set[Side] = set([])
        for road in self.river:
            sides.add(road.a)
            sides.add(road.b)
        return set(sides)

    def get_city_sides(self) -> Set[Side]:
        sides: Set[Side] = set([])
        for side_list in self.city:
            for side in side_list:
                sides.add(side)
        return set(sides)

    def has_river(self) -> bool:
        return len(self.river) > 0

    def get_type(self, side: Side):
        if self.unplayable_sides.__contains__(side):
            return TerrainType.UNPLAYABLE

        if side == Side.CENTER and self.chapel_or_flowers:
            return TerrainType.CHAPEL_OR_FLOWERS

        if self.get_river_ends().__contains__(side):
            return TerrainType.UNPLAYABLE

        if self.get_road_ends().__contains__(side):
            return TerrainType.ROAD

        if self.get_city_sides().__contains__(side):
            return TerrainType.CITY

        if self.grass.__contains__(side):
            return TerrainType.GRASS

    def to_json(self):
        return {
            "description": self.description,
            "river": list(map(lambda x: x.to_json(), self.river)),
            "road": list(map(lambda x: x.to_json(), self.road)),
            "city": list(map(lambda x: x.to_json(), self.city)),
            "grass": list(map(lambda x: x.to_json(), self.grass)),
            "shield": self.shield,
            "chapel_or_flowers": self.chapel_or_flowers,
            "inn": list(map(lambda x: x.to_json(), self.inn)),
            "unplayable_sides": list(map(lambda x: x.to_json(), self.unplayable_sides))
        }

    def __str__(self):
        return json.dumps(self.to_json(), indent=2)

    def turn(self, times: int):
        return Tile(
            description=self.description,
            turns=times,
            road=list(map(lambda x: turn_connection(x, times), self.road)),
            river=list(map(lambda x: turn_connection(x, times), self.river)),
            city=list(map(lambda x: turn_sides(x, times), self.city)),
            grass=list(map(lambda x: turn_side(x, times), self.grass)),
            shield=self.shield,
            chapel_or_flowers=self.chapel_or_flowers,
            inn=list(map(lambda x: turn_side(x, times), self.inn)),
            unplayable_sides=list(map(lambda x: turn_side(x, times), self.unplayable_sides)),
            image=self.image
        )
