from core.property import Pos_x, Pos_y, Shape_x, Shape_y

class Event:
    pass

class NoTargetEvent(Event):

    def __init__(self, name):
        self.name = name
    
    def copy(self):
        raise NotImplementedError("Subclasses must implement this method.")
    
    def __repr__(self):
        raise NotImplementedError("Subclasses must implement this method.")

    @staticmethod
    def check() -> bool:
        return False

class GlobalEvent(NoTargetEvent):

    def __init__(self, name):
        self.name = name
    
    def copy(self): return GlobalEvent(self.name)
    
    def __repr__(self):
        return self.name

    @staticmethod
    def check() -> bool:
        return True

class CommandEvent(NoTargetEvent):

    def __init__(self, name):
        self.name = name
    
    def copy(self): return CommandEvent(self.name)
    
    def __repr__(self):
        return self.name

    @staticmethod
    def check() -> bool:
        return True

class SingleTargetEvent(Event):

    @staticmethod
    def check(previous, current, current_others) -> bool:
        raise NotImplementedError("Subclasses must implement this method.")

class Contact(Event):

    @staticmethod
    def check(previous, current, current_others) -> bool:
        raise NotImplementedError("Subclasses must implement this method.")

class Contact_With_Something_T(Contact):

    @staticmethod
    def check(previous, current, current_others) -> bool:
        hb1 = ((current.properties[Pos_x] - current.properties[Shape_x], current.properties[Pos_y] - current.properties[Shape_y]), (current.properties[Pos_x] + current.properties[Shape_x], current.properties[Pos_y] + current.properties[Shape_y]))
        for oe in current_others:
            hb2 = ((oe.properties[Pos_x] - oe.properties[Shape_x], oe.properties[Pos_y] - oe.properties[Shape_y]), (oe.properties[Pos_x] + oe.properties[Shape_x], oe.properties[Pos_y] + oe.properties[Shape_y]))
            if (hb1[0][1] - 1 == hb2[1][1] and hb1[0][0] <= hb2[1][0] and hb1[1][0] >= hb2[0][0]):
                return True
        return False

class Contact_With_Something_B(Contact):

    @staticmethod
    def check(previous, current, current_others) -> bool:
        hb1 = ((current.properties[Pos_x] - current.properties[Shape_x], current.properties[Pos_y] - current.properties[Shape_y]), (current.properties[Pos_x] + current.properties[Shape_x], current.properties[Pos_y] + current.properties[Shape_y]))
        for oe in current_others:
            hb2 = ((oe.properties[Pos_x] - oe.properties[Shape_x], oe.properties[Pos_y] - oe.properties[Shape_y]), (oe.properties[Pos_x] + oe.properties[Shape_x], oe.properties[Pos_y] + oe.properties[Shape_y]))
            if (hb1[1][1] + 1 == hb2[0][1] and hb1[0][0] <= hb2[1][0] and hb1[1][0] >= hb2[0][0]):
                #print(f'contact between bottom side of {current}({hb1}) and {oe}({hb2})')
                return True
        return False

class Contact_With_Something_L(Contact):

    @staticmethod
    def check(previous, current, current_others) -> bool:
        hb1 = ((current.properties[Pos_x] - current.properties[Shape_x], current.properties[Pos_y] - current.properties[Shape_y]), (current.properties[Pos_x] + current.properties[Shape_x], current.properties[Pos_y] + current.properties[Shape_y]))
        for oe in current_others:
            hb2 = ((oe.properties[Pos_x] - oe.properties[Shape_x], oe.properties[Pos_y] - oe.properties[Shape_y]), (oe.properties[Pos_x] + oe.properties[Shape_x], oe.properties[Pos_y] + oe.properties[Shape_y]))
            if (hb1[0][0] - 1 == hb2[1][0] and hb1[0][1] <= hb2[1][1] and hb1[1][1] >= hb2[0][1]):
                return True
        return False

class Contact_With_Something_R(Contact):

    @staticmethod
    def check(previous, current, current_others) -> bool:
        hb1 = ((current.properties[Pos_x] - current.properties[Shape_x], current.properties[Pos_y] - current.properties[Shape_y]), (current.properties[Pos_x] + current.properties[Shape_x], current.properties[Pos_y] + current.properties[Shape_y]))
        for oe in current_others:
            hb2 = ((oe.properties[Pos_x] - oe.properties[Shape_x], oe.properties[Pos_y] - oe.properties[Shape_y]), (oe.properties[Pos_x] + oe.properties[Shape_x], oe.properties[Pos_y] + oe.properties[Shape_y]))
            if (hb1[1][0] + 1 == hb2[0][0] and hb1[0][1] <= hb2[1][1] and hb1[1][1] >= hb2[0][1]):
                return True
        return False

event_pool = [Contact_With_Something_T, Contact_With_Something_B, Contact_With_Something_L, Contact_With_Something_R]
