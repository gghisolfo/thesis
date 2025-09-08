

class ID_generator:

    def __init__(self, init_id= -1):
        self.prev_id = init_id

    def __call__(self) -> int:
        self.prev_id += 1
        return self.prev_id
    
    def copy(self): return ID_generator(self.prev_id)


def compute_diff(pred, patch):

    diff = 0

    for property_class, value in patch.properties.items():
        diff += abs(pred[property_class] - value)

    return diff

def equal_collections(list1, list2):
    if len(list1) != len(list2):
        return False
    for obj1 in list1:
        if not any(obj1 == obj2 for obj2 in list2):
            return False
    return True
