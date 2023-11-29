from wingedsheep.carcassonne.objects.actions.action import Action
from wingedsheep.carcassonne.objects.coordinate import Coordinate
from wingedsheep.carcassonne.objects.tile import Tile


class TileAction(Action):
    def __init__(self, tile: Tile, coordinate: Coordinate, tile_rotations: int):
        self.tile = tile
        self.coordinate = coordinate
        self.tile_rotations = tile_rotations

    def __eq__(self, other):
        if isinstance(other, TileAction):
            return self.coordinate == other.coordinate and self.tile_rotations == other.tile_rotations
        return False

    def __str__(self):
        return "Coordinate: " + self.coordinate.__str__() + "Rotation: " + str(self.tile_rotations)