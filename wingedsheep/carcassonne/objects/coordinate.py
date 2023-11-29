class Coordinate:

    def __init__(self, row: int, column: int):
        self.row = row
        self.column = column

    def __eq__(self, other):
        return self.row == other.row and self.column == other.column

    def __str__(self):
        attributes = ""
        for attribute, value in vars(self).items():
            attributes += f"{attribute}: {value}\n"
        return attributes

    def __hash__(self):
        return hash((self.row, self.column))
