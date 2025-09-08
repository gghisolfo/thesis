

class Property:

    @staticmethod
    def compute(previous, current):
        raise NotImplementedError()

    @staticmethod
    def effects(property_values) -> dict:
        raise NotImplementedError()
    
    @staticmethod
    def dependencies():
        raise NotImplementedError()
    
    @staticmethod
    def name():
        raise NotImplementedError()

class Pos_x(Property):

    @staticmethod
    def compute(previous, current): return current.properties[Pos_x]
    
    @staticmethod
    def effects(properties): return {}
    
    @staticmethod
    def dependencies(): return []
    
    @staticmethod
    def name(): return 'pos_x'

class Pos_y(Property):

    @staticmethod
    def compute(previous, current): return current.properties[Pos_y]
    
    @staticmethod
    def effects(properties): return {}
    
    @staticmethod
    def dependencies(): return []
    
    @staticmethod
    def name(): return 'pos_y'

class Shape_x(Property):

    @staticmethod
    def compute(previous, current): return current.properties[Shape_x]
    
    @staticmethod
    def effects(properties): return {}
    
    @staticmethod
    def dependencies(): return []
    
    @staticmethod
    def name(): return 'shape_x'

class Shape_y(Property):

    @staticmethod
    def compute(previous, current): return current.properties[Shape_y]
    
    @staticmethod
    def effects(properties): return {}
    
    @staticmethod
    def dependencies(): return []
    
    @staticmethod
    def name(): return 'shape_y'

class Speed_x(Property):

    @staticmethod
    def compute(previous, current): return current.properties[Pos_x] - previous.properties[Pos_x]
    
    @staticmethod
    def effects(properties):
        return {Pos_x: properties[Speed_x]}
    
    @staticmethod
    def dependencies(): return [Pos_x]
    
    @staticmethod
    def name(): return 'vx'

class Speed_y(Property):

    @staticmethod
    def compute(previous, current): return current.properties[Pos_y] - previous.properties[Pos_y]
    
    @staticmethod
    def effects(properties):
        return {Pos_y: properties[Speed_y]}
    
    @staticmethod
    def dependencies(): return [Pos_y]
    
    @staticmethod
    def name(): return 'vy'