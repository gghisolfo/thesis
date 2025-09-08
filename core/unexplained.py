from core.property import Speed_x, Speed_y
from core.patch import Patch
from core.events import Event, GlobalEvent, CommandEvent

class Phenomenon:

    def __init__(self, info):
        pass

    def test(self, phenomenon):
        pass

    def __repr__(self):
        pass

    def my_hash(self):
        pass

class SpecificUnexplainedPhenomenon(Phenomenon):

    def __init__(self, info):
        self.unexplained_class = info['unexplained_class']

    def test(self, phenomenon):
        return isinstance(phenomenon, self.unexplained_class)
    
    def __repr__(self):
        return self.unexplained_class.__name__
    
    def __eq__(self, other):
        if isinstance(other, SpecificUnexplainedPhenomenon): return self.unexplained_class == other.unexplained_class
        else: return False

    def my_hash(self):
        return self.unexplained_class.__name__

class NumericalUnexplainedPhenomenon(Phenomenon):

    def __init__(self, info):
        self.property_class = info['property_class']
        self.a = info['a']
        self.b = info['b']

    def test(self, phenomenon):
        if not isinstance(phenomenon, PropertyChange): return False
        if self.property_class != phenomenon.property_class: return False
        return (phenomenon.final_value == self.a * phenomenon.previous_value + self.b)
    
    def __repr__(self):
        return f'{self.property_class.name()}(i+1) = {self.a} * {self.property_class.name()}(i) + {self.b}'
    
    def __eq__(self, other):
        if isinstance(other, NumericalUnexplainedPhenomenon): return (self.property_class == other.property_class and self.a == other.a and self.b == other.b)
        else: return False

    def my_hash(self):
        return f'{self.property_class}{self.a}{self.b}'

class EventPhenomenon(Phenomenon):

    def __init__(self, info):
        self.event_class = info['event_class']

    def test(self, phenomenon):
        if isinstance(phenomenon, EventPhenomenon): return self.event_class == phenomenon.event_class
        if isinstance(phenomenon, type):
            if issubclass(phenomenon, Event): return phenomenon is self.event_class
        return False
    
    def __eq__(self, other):
        if isinstance(other, EventPhenomenon): return self.event_class == other.event_class
        else: return False
    
    def __repr__(self):
        return self.event_class.__name__

    def my_hash(self):
        return self.event_class.__name__

class GlobalEventPhenomenon(Phenomenon):

    def __init__(self, info):
        self.name = info['name']

    def test(self, phenomenon):
        if isinstance(phenomenon, GlobalEventPhenomenon): return self.name == phenomenon.name
        if isinstance(phenomenon, GlobalEvent) or isinstance(phenomenon, CommandEvent): return self.name == phenomenon.name
        if isinstance(phenomenon, str): return phenomenon == self.name
        return False
    
    def __eq__(self, other):
        if isinstance(other, GlobalEventPhenomenon): return self.name == other.name
        else: return False
    
    def __repr__(self):
        return self.name

    def my_hash(self):
        return self.name

class UnexplainedChange:
    pass

class UnexplainedNumericalChange(UnexplainedChange):
    pass

class UnexplainedSpecificChange(UnexplainedChange):
    pass

class PropertyChange(UnexplainedNumericalChange):

    def __init__(self, property_class, previous_value, final_value):
        self.property_class = property_class
        self.previous_value = previous_value
        self.final_value = final_value

    def copy(self):
        return PropertyChange(self.property_class, self.previous_value, self.final_value)
    
    def __repr__(self): return f'PropertyChange({self.property_class.name()}: {self.previous_value} -> {self.final_value})'

    def __eq__(self, other):
        if isinstance(other, PropertyChange): return self.property_class == other.property_class and self.final_value - self.previous_value == other.final_value - other.previous_value
        else: return False

    def my_hash(self):
        return ('PropertyChange', self.property_class, self.final_value - self.previous_value)

class Appearance(UnexplainedSpecificChange):

    def __init__(self):
        pass

    def copy(self):
        return Appearance()
    
    def __repr__(self): return 'Appearance'

    def __eq__(self, other):
        if isinstance(other, Appearance): return True
        else: return False

    def my_hash(self):
        return ('Appearance', None, None)

class Disappearance(UnexplainedSpecificChange):

    def __init__(self):
        pass

    def copy(self):
        return Disappearance()

    def __eq__(self, other):
        if isinstance(other, Disappearance): return True
        else: return False

    def my_hash(self):
        return ('Disappearance', None, None)
    
    def __repr__(self): return 'Disappearance'

class Duplication(UnexplainedSpecificChange):

    def __init__(self, from_obj):
        self.from_obj = from_obj

    def copy(self):
        return Duplication(self.from_obj)
    
    def __repr__(self): return f'Duplication(from {self.from_obj})'

    def __eq__(self, other):
        if isinstance(other, Duplication): return self.from_obj == other.from_obj

    def my_hash(self):
        return ('Duplication', self.from_obj.id, None) # this is wrong and just a placeholder

# change the dict keys to string
# define Property as mother class and Property_0 and Property_1
# Property_0's compute just return the new patch's property associated with the string of reference (starting with only Property_0 and in future expanding to Property_1)
# Property_1's compute compute the actual value change between the last two frames
# Property_0's effect return the same value
# Property_1's effect return the change and in which property
# then modify the function to test possible combinations Property_1's changes that could explain the diff and return all possible list of unexplaineds
def check_for_speed(obj, patch, frame_id, next_patches):

    ##
    #return False, False, None, None
    ##

    last_patch = obj.sequence[-1]
    current_properties = {fid: {k: v for k, v in prop.items()} for fid, prop in obj.properties.items()}
    current_properties[frame_id - 1][Speed_x] = Speed_x.compute(last_patch, patch)
    current_properties[frame_id - 1][Speed_y] = Speed_y.compute(last_patch, patch)

    dummy_object = obj.create_dummy([obj.frames_id[-1]], [last_patch], current_properties, obj.rules)

    all_ok = True
    for property_class, value in patch.properties.items():
        if dummy_object.prediction[property_class] != value:
            all_ok = False
            break

    if not all_ok: return False, False, None, None

    unexplained_dict = {}

    current_properties[frame_id] = dummy_object.prediction

    q1_unexplained = []
    if Speed_x in obj.properties[frame_id - 1].keys():
        if obj.properties[frame_id - 1][Speed_x] != current_properties[frame_id][Speed_x]:
            q1_unexplained.append(PropertyChange(Speed_x, obj.properties[frame_id - 1][Speed_x], current_properties[frame_id][Speed_x]))
    elif current_properties[frame_id][Speed_x] != 0:
        q1_unexplained.append(PropertyChange(Speed_x, 0, current_properties[frame_id][Speed_x]))
    if Speed_y in obj.properties[frame_id - 1].keys():
        if obj.properties[frame_id - 1][Speed_y] != current_properties[frame_id][Speed_y]:
            q1_unexplained.append(PropertyChange(Speed_y, obj.properties[frame_id - 1][Speed_y], current_properties[frame_id][Speed_y]))
    elif current_properties[frame_id][Speed_y] != 0:
        q1_unexplained.append(PropertyChange(Speed_y, 0, current_properties[frame_id][Speed_y]))

    if not q1_unexplained: return False, False, None, None

    if frame_id - 1 in unexplained_dict.keys(): unexplained_dict[frame_id - 1].extend(q1_unexplained)
    else: unexplained_dict[frame_id - 1] = q1_unexplained

    ##
    return True, False, unexplained_dict, current_properties
    ##

    dummy_object = obj.create_dummy([frame_id], [patch], current_properties, obj.rules)
    new_pred = dummy_object.prediction

    confirmed = False
    if next_patches:
        for np in next_patches:
            all_ok = True
            for property_class, value in np.properties.items():
                if new_pred[property_class] != value:
                    all_ok = False
                    break
            if all_ok:
                confirmed = True
                break
    else: confirmed = True

    if confirmed: return True, True, unexplained_dict, current_properties
    else: return True, False, unexplained_dict, current_properties

def check_for_property0_changes(obj, patch, frame_id):

    unexplained = []
    current_properties = {fid: {k: v for k, v in prop.items()} for fid, prop in obj.properties.items()}
    current_properties[frame_id] = obj.prediction

    for property_class, value in patch.properties.items():
        if current_properties[frame_id][property_class] != value:
            unexplained.append(PropertyChange(property_class, obj.properties[frame_id - 1][property_class], value))
            current_properties[frame_id][property_class] = value

    if unexplained: return True, {frame_id: unexplained}, current_properties
    else: return False, {}, current_properties


def check_disappearance(obj, frame_id):
    return True, {frame_id: [Disappearance()]}, obj.properties[obj.frames_id[-1]]

# same for this one
def check_multiple_holes_simple(obj, patch, frame_id):

    starting_frame_id = obj.frames_id[-1]
    dummy_object = obj.create_dummy([obj.frames_id[-1]], [obj.sequence[-1]], obj.properties, obj.rules)

    for i in range(starting_frame_id + 1, frame_id):

        dummy_object.update(i, Patch('dummy', dummy_object.prediction), dummy_object.prediction, [])

    all_ok = True
    for property_class, value in patch.properties.items():
        if dummy_object.prediction[property_class] != value:
            all_ok = False
            break

    if all_ok: return True, {frame_id: [Appearance(frame_id)]}, dummy_object.prediction
    else: return False, None, None

# same for this one
def check_multiple_holes_speed(obj, patch, frame_id):

    dummy_object = obj.create_dummy([obj.frames_id[-1]], [obj.sequence[-1]], obj.properties, obj.rules)

    starting_frame_id = obj.frames_id[-1]
    last_patch = obj.sequence[-1]
    last_properties = {k: v for k, v in obj.properties[obj.frames_id[-1]].items()}
    last_properties[Speed_x] = Speed_x.compute(last_patch, patch) / (frame_id - starting_frame_id)
    last_properties[Speed_y] = Speed_y.compute(last_patch, patch) / (frame_id - starting_frame_id)

    for i in range(starting_frame_id + 1, frame_id):

        dummy_object.update(i, Patch('dummy', dummy_object.prediction), dummy_object.prediction, [])

    all_ok = True
    for property_class, value in patch.properties.items():
        if dummy_object.prediction[property_class] != value:
            all_ok = False
            break

    if all_ok: return True, {starting_frame_id: [PropertyChange(Speed_x, obj.properties[obj.frames_id[-1]][Speed_x], dummy_object.prediction[Speed_x]), PropertyChange(Speed_y, obj.properties[obj.frames_id[-1]][Speed_y], dummy_object.prediction[Speed_y])], frame_id: Appearance()}, dummy_object.prediction
    else: return False, None, None


# same
def check_blink(obj, patch, frame_id):

    last_properties = {k: v for k, v in obj.properties[obj.frames_id[-1]].items()}

    for property_class, value in patch.properties.items():
        last_properties[property_class] = value

    return True, {frame_id: [Disappearance(), Appearance()]}, last_properties

# same
def check_duplication(obj, patch, frame_id):

    last_properties = {k: v for k, v in obj.properties[obj.frames_id[-1]].items()}

    for property_class, value in patch.properties.items():
        last_properties[property_class] = value

    return True, {frame_id: [Duplication(obj)]}, last_properties