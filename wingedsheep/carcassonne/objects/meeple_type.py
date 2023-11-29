from enum import Enum, unique

@unique
class MeepleType(Enum):
    NORMAL = "normal"
    ABBOT = "abbot"
    FARMER = "farmer"
    BIG = "big"
    BIG_FARMER = "big_farmer"

    def to_json(self):
        return self.value

    def __str__(self):
        return self.value

    def __eq__(self, other):
        return self.value == other.value

    def __hash__(self):
        return hash(self.value)