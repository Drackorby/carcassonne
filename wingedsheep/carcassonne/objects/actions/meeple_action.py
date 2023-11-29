from wingedsheep.carcassonne.objects.actions.action import Action
from wingedsheep.carcassonne.objects.coordinate_with_side import CoordinateWithSide
from wingedsheep.carcassonne.objects.meeple_type import MeepleType


class MeepleAction(Action):
    def __init__(self, meeple_type: MeepleType, coordinate_with_side: CoordinateWithSide, remove: bool = False):
        self.meeple_type = meeple_type
        self.coordinate_with_side = coordinate_with_side
        self.remove = remove

    def __eq__(self, other):
        if isinstance(other, MeepleAction):
            return self.meeple_type == other.meeple_type and self.coordinate_with_side == other.coordinate_with_side and self.remove == other.remove
        return False
