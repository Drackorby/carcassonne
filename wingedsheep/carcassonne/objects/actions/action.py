class Action:

    def print_attributes(self):
        for attribute, value in vars(self).items():
            print(f"{attribute}: {value}")

    def __eq__(self, other):
        if isinstance(other, Action):
            return True
        return False
