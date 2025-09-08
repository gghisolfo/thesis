from core.property import Pos_x, Pos_y

class Patch:
    def __init__(self, description, properties):
        self.description = description # debug only
        self.properties = {k: v for k, v in properties.items()}

    def __repr__(self):
        return f"Patch({self.description}, {[(k.name(), v) for k, v in self.properties.items()]})"
