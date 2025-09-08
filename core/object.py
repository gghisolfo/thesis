from core.unexplained import NumericalUnexplainedPhenomenon, PropertyChange
from utils import equal_collections
from core.events import event_pool

class Object:

    def __init__(self, frames_id, sequence, properties= {}, unexplained= {}, explained_unexplained= {}, events= {}, global_events= {}, rules= []):

        self.frames_id = frames_id[:]

        self.sequence = sequence[:]

        self.properties = {fid: {k: v for k, v in prop.items()} for fid, prop in properties.items()}

        self.unexplained = {fid: [ex.copy() for ex in v_list] for fid, v_list in unexplained.items()}

        self.explained_unexplained = {fid: [ex.copy() for ex in v_list] for fid, v_list in explained_unexplained.items()}

        self.events = {fid: [ev for ev in v_list] for fid, v_list in events.items()}

        self.global_events = {fid: [ev.copy() for ev in v_list] for fid, v_list in global_events.items()}

        self.rules = rules[:]

        self.prediction, self.predicted_explained = self.compute_next(frames_id[-1])

    def reset_explained_and_rules(self):
        self.rules = []
        new_unexplained = {}
        for fid in list(self.unexplained.keys()) + list(self.explained_unexplained.keys()):
            fid_unexplained = []
            if fid in self.unexplained.keys(): fid_unexplained.extend(self.unexplained[fid])
            if fid in self.explained_unexplained.keys(): fid_unexplained.extend(self.explained_unexplained[fid])
            new_unexplained[fid] = fid_unexplained
        self.unexplained = new_unexplained
        self.explained_unexplained = {}

    def copy(self):
        return Object(self.frames_id, self.sequence, self.properties, self.unexplained, self.explained_unexplained, self.events, self.global_events, self.rules)
    
    def create_dummy(self, frames_id, sequence, properties, rules):
        return Object(frames_id, sequence, properties, {}, {}, {}, {}, rules)

    def compute_next(self, frame_id):

        new_properties = {property_class: value for property_class, value in self.properties[self.frames_id[-1]].items()}

        predicted_explained = []
        for rule in self.rules:
            triggered, effects, _ = rule.trigger(self, frame_id - rule.cause_offset)
            if triggered:
                for effect in effects:
                    if isinstance(effect, NumericalUnexplainedPhenomenon):
                        p_class = effect.property_class
                        if p_class not in new_properties.keys():
                            new_properties[p_class] = 0
                            previous_value = 0
                        else: previous_value = new_properties[p_class]
                        new_properties[p_class] = effect.a * new_properties[p_class] + effect.b
                        predicted_explained.append(PropertyChange(p_class, previous_value, new_properties[p_class]))

        new_new_properties = {property_class: value for property_class, value in new_properties.items()}

        for property_class in new_properties.keys():
            for prop_to_modify, change in property_class.effects(new_properties).items():
                new_new_properties[prop_to_modify] += change

        return new_new_properties, predicted_explained
    
    def update(self, frame_id, patch, new_properties, other_patches, global_events):

        self.frames_id.append(frame_id)
        self.sequence.append(patch)
        self.properties = {fid: {k: v for k, v in prop.items()} for fid, prop in new_properties.items()}

        previous_patch = self.sequence[-2]
        current_patch = self.sequence[-1]

        new_events = []
        for event in event_pool:
            if event.check(previous_patch, current_patch, other_patches):
                new_events.append(event)
        self.events[frame_id] = new_events

        self.global_events[frame_id] = [ev.copy() for ev in global_events]

        self.prediction, self.predicted_explained = self.compute_next(frame_id)

    def add_unexplained(self, unexplained_dict):
        for frame_id, unexplained in unexplained_dict.items():
            if frame_id in self.unexplained.keys(): self.unexplained[frame_id].extend([ex.copy() for ex in unexplained])
            else: self.unexplained[frame_id] = [ex.copy() for ex in unexplained]

    def add_explained(self, explained_dict):
        for frame_id, explained in explained_dict.items():
            if frame_id in self.explained_unexplained.keys(): self.explained_unexplained[frame_id].extend([ex.copy() for ex in explained])
            else: self.explained_unexplained[frame_id] = [ex.copy() for ex in explained]

    def add_rule(self, rule): self.rules.append(rule)

    def __eq__(self, other):
        
        if isinstance(other, Object):
            if set(self.frames_id) != set(other.frames_id): return False
            if not equal_collections(self.sequence, other.sequence): return False
            #if not equal_collections(self.unexplained, other.unexplained): return False
            if not equal_collections(self.events, other.events): return False
            if not equal_collections(self.rules, other.rules): return False
            return True

    def __repr__(self):
        ss = f'[{self.sequence[0].description}'
        for patch in self.sequence[1:]:
            ss += f', {patch.description}'
        ss += ']\nlast properties: {'
        for prop_class, val in self.properties[self.frames_id[-1]].items():
            ss += f'({prop_class.name()}: {val})'
        ss += '}\nunexplaineds: {'
        for frame_id, unexpl in self.unexplained.items():
            ss += f'{frame_id}: {unexpl} |'
        ss += '}\nexplained unexplaineds: {'
        for frame_id, expl in self.explained_unexplained.items():
            ss += f'{frame_id}: {expl} |'
        ss += '}\nevents: {'
        for frame_id, ev in self.events.items():
            ss += f'{frame_id}: {ev} |'
        ss += '}\nglobal events: {'
        for frame_id, ev in self.global_events.items():
            ss += f'{frame_id}: {ev} |'
        ss += '}\nrules: {'
        for ruid, ru in enumerate(self.rules):
            ss += f'\trule_{ruid}: {ru.my_hash()} |'
        ss += '}'
        return ss