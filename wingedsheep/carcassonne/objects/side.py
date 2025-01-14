from enum import Enum, unique


@unique
class Side(Enum):
    TOP = "top"
    RIGHT = "right"
    BOTTOM = "bottom"
    LEFT = "left"
    CENTER = "center"

    # special for farmers
    TOP_LEFT = "top_left"
    TOP_RIGHT = "top_right"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM_RIGHT = "bottom_right"

    def to_json(self):
        return self.value

    def __str__(self):
        return self.value

    def __eq__(self, other):
        return self.value == other.value

    def __hash__(self):
        return hash(self.value)

