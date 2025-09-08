from utils.various import equal_collections
from core.unexplained import (
    UnexplainedSpecificChange,
    UnexplainedNumericalChange,
    SpecificUnexplainedPhenomenon,
    NumericalUnexplainedPhenomenon,
    EventPhenomenon,
    GlobalEventPhenomenon,
    )
from core.events import Event, CommandEvent, GlobalEvent

RULE_MIN_TIMES = 1
CAUSE_EFFECT_MAX_OFFSET = 3

class Rule:

    def __init__(self, cause_offset, causes, effects):
        self.cause_offset = cause_offset  # Difference between effect frame and cause frame
        self.causes = causes[:]           # List of generalized causes as phenomenons
        self.effects = effects[:]         # List of effects as phenomenons

    def trigger(self, obj, frame_id, debug= False):
        if debug: print(f'\ntesting {obj.sequence[0].description} at frame {frame_id}')
        possible_causes = []
        possible_causes += obj.unexplained[frame_id] if frame_id in obj.unexplained.keys() else []
        possible_causes += obj.explained_unexplained[frame_id] if frame_id in obj.explained_unexplained.keys() else []
        possible_causes += obj.events[frame_id] if frame_id in obj.events.keys() else []
        possible_causes += obj.global_events[frame_id] if frame_id in obj.global_events.keys() else []
        if debug: print(f'possible_causes: {possible_causes}')
        if debug:
            for cause in self.causes:
                for pc in possible_causes:
                    print(f'cause_class: {type(cause)} - pc_class: {type(pc)}')
                    print(f'{cause}.test({pc}) -> {cause.test(pc)}')
        if all([any([cause.test(pc) for pc in possible_causes]) for cause in self.causes]):
            return True, self.effects, self.cause_offset
        else: return False, None, None

    def __eq__(self, other):
        if not (isinstance(other, Rule)): return False
        if self.cause_offset != other.cause_offset: return False
        if not equal_collections(self.causes, other.causes): return False
        if not equal_collections(self.effects, other.effects): return False
        return True
    
    def my_hash(self):
        ss = f'{self.cause_offset}'
        for cause_hash in sorted([cause.my_hash() for cause in self.causes]):
            ss += cause_hash
        for effect_hash in sorted([effect.my_hash() for effect in self.effects]):
            ss += effect_hash
        return ss

    def __repr__(self):
        return f'rule\nwith causes: {self.causes}\nwith effects: {self.effects}\nafter {self.cause_offset} frames'

A_RANGE = [-1, 0, 1, 2, -2]#, 3, -3, 4, -4, 5, -5, 6, -6]
B_RANGE = [-1, 0, 1]

def convert_to_phenomenon(event_or_unexplained):

    if isinstance(event_or_unexplained, UnexplainedSpecificChange):
        return [SpecificUnexplainedPhenomenon({'unexplained_class': event_or_unexplained.__class__})], 1
    elif isinstance(event_or_unexplained, UnexplainedNumericalChange):
        #print(event_or_unexplained)
        previous = event_or_unexplained.previous_value
        final = event_or_unexplained.final_value
        #print(f'{previous} -> {final}')
        if previous == 0:
            #print('previous == 0')
            a = 0
            b = final
            phenoms = [NumericalUnexplainedPhenomenon({'a': a, 'b': b, 'property_class': event_or_unexplained.property_class})]
            #print(f'a: {a}, b: {b}')
        elif final == 0:
            #print('final == 0')
            a = 0
            b = 0
            phenoms = [NumericalUnexplainedPhenomenon({'a': a, 'b': b, 'property_class': event_or_unexplained.property_class})]
            #print(f'a: {a}, b: {b}')
        elif final % previous == 0:
            #print('final // previous == 0')
            a = final // previous
            b = 0
            phenoms = [NumericalUnexplainedPhenomenon({'a': a, 'b': b, 'property_class': event_or_unexplained.property_class})]
            #print(f'a: {a}, b: {b}')
        else:
            #print('else')
            a1 = 1
            b1 = final - previous
            #a2 = 0
            #b2 = final
            phenoms = [NumericalUnexplainedPhenomenon({'a': a1, 'b': b1, 'property_class': event_or_unexplained.property_class}),
                       #NumericalUnexplainedPhenomenon({'a': a2, 'b': b2, 'property_class': event_or_unexplained.property_class}),
                       ]
            #print(f'a1: {a1}, b1: {b1}')
            #print(f'a2: {a2}, b2: {b2}')
        return phenoms, 1
    elif isinstance(event_or_unexplained, CommandEvent) or isinstance(event_or_unexplained, GlobalEvent):
        return [GlobalEventPhenomenon({'name': event_or_unexplained.__repr__()})], 0
    elif issubclass(event_or_unexplained, Event):
        return [EventPhenomenon({'event_class': event_or_unexplained}), EventPhenomenon({'event_class': event_or_unexplained.__base__})], 0
    else:
        print('nah2')
        exit(0)

def new_infer_rules(ind, present_objects, not_present_objects, frame_id, debug= False):

    explained_score = 0

    seen_rules = []

    all_ind_objs = {obj_id: present_objects[obj_id] if obj_id in present_objects.keys() else not_present_objects[obj_id] for obj_id in ind}

    for obj in all_ind_objs.values():

        if 'ball' in obj.sequence[0].description and frame_id == 1400000: debug = True

        obj.reset_explained_and_rules()

        obj_cause_effect_offset_times = {}
        obj_cause_effect_offset_rule = {}
        obj_cause_effect_offset_fid = {}
        obj_cause_times = {}

        all_possible_causes = {}
        frames_with_possible_causes = []
        for ffid in range(0, frame_id + 1):
            all_possible_causes_ffid = []
            if ffid in obj.unexplained.keys(): all_possible_causes_ffid.extend(obj.unexplained[ffid])
            if ffid in obj.events.keys(): all_possible_causes_ffid.extend(obj.events[ffid])
            if ffid in obj.global_events.keys(): all_possible_causes_ffid.extend(obj.global_events[ffid])
            if all_possible_causes_ffid:
                all_possible_causes[ffid] = all_possible_causes_ffid
                frames_with_possible_causes.append(ffid)


        for cf in range(0, frame_id + 1):
            #if debug: print(f'cf: {cf}')

            if cf in frames_with_possible_causes:

                for ev in all_possible_causes[cf]:
                    causes, starting_offset = convert_to_phenomenon(ev)
                    for cause in causes:
                        cause_hash = cause.my_hash()
                        #if debug: print(f'\t\tcause: {cause_hash}')

                        if cause_hash in obj_cause_times.keys(): obj_cause_times[cause_hash] += 1
                        else: obj_cause_times[cause_hash] = 1

                        for ef in range(cf + starting_offset, cf + CAUSE_EFFECT_MAX_OFFSET if cf + CAUSE_EFFECT_MAX_OFFSET <= frame_id else frame_id + 1):
                            #if debug: print(f'\tef: {ef}')
                            offset = ef - cf

                            effects_set = obj.unexplained | obj.explained_unexplained
                            if ef in effects_set.keys():
                                for un in effects_set[ef]:
                                    effects, _ = convert_to_phenomenon(un)
                                    for effect in effects:
                                        effect_hash = effect.my_hash()
                                        #if debug: print(f'\t\t\teffect: {effect}')
                                        if cause != effect:

                                            rule = Rule(offset, [cause], [effect])

                                            if (cause_hash, effect_hash, offset) in obj_cause_effect_offset_times.keys():
                                                obj_cause_effect_offset_times[((cause_hash, effect_hash, offset))] += 1
                                                obj_cause_effect_offset_fid[((cause_hash, effect_hash, offset))].append(ef)
                                            else:
                                                obj_cause_effect_offset_times[((cause_hash, effect_hash, offset))] = 1
                                                obj_cause_effect_offset_rule[((cause_hash, effect_hash, offset))] = rule
                                                obj_cause_effect_offset_fid[((cause_hash, effect_hash, offset))] = [ef]

        if debug:
            print('-')
            print(obj.sequence[0].description)
            print(obj.events)
            print(obj.global_events)
            print(obj.unexplained)
            print('\nobj_cause_effect_offset_times\n')
            for x, y in obj_cause_effect_offset_times.items(): print(f'{x}: {y}')
            print('\nobj_cause_times\n')
            for x, y in obj_cause_times.items(): print(f'{x}: {y}')

        potential_rules = []
        cause_classes = []
        rule_times = []

        obj_explained_score = 0

        for (cause_hash, effect_hash, offset), times in obj_cause_effect_offset_times.items():
            #if times == obj_cause_times[cause_hash]:
            if times >= obj_cause_times[cause_hash] - 1:
                #obj_explained_score -= times
                rule = obj_cause_effect_offset_rule[((cause_hash, effect_hash, offset))]
                #rule_hash = rule.my_hash()

                #if rule_hash not in seen_rules: seen_rules.append(rule_hash)

                potential_rules.append(rule)
                rule_times.append(times)

                if isinstance(rule.causes[0], EventPhenomenon): cause_classes.append(rule.causes[0].event_class)
                else: cause_classes.append(None)

        if debug:
            print('\n-\n')
            for rule_pid, rule in enumerate(potential_rules): print(f'potential_rule_{rule_pid}: {rule.my_hash()}')
        
        #if debug: print('\n-\n\npotential_rule_selection')
        new_potential_rules = []
        for pr, cc in zip(potential_rules, cause_classes):
            #if debug:
            #    print('-')
            #    print(f'pr: {pr.my_hash()}')
            #    print(f'cc: {cc}')
            if cc is not None:
                if cc.__base__ not in cause_classes: new_potential_rules.append(pr)
            else: new_potential_rules.append(pr)
        
        if debug:
            print('\n-\n')
            for rule_pid, rule in enumerate(new_potential_rules): print(f'New_potential_rule{rule_pid}: {rule.my_hash()}')

        effect_frame = {}
        for rule in new_potential_rules:
            cause_hash = rule.causes[0].my_hash()
            effect_hash = rule.effects[0].my_hash()
            for effect_fid in obj_cause_effect_offset_fid[(cause_hash, effect_hash, rule.cause_offset)]:
                if (effect_hash, effect_fid) in effect_frame.keys():
                    effect_frame[(effect_hash, effect_fid)].append((obj_cause_effect_offset_rule[(cause_hash, effect_hash, rule.cause_offset)], obj_cause_effect_offset_times[(cause_hash, effect_hash, rule.cause_offset)]))
                else:
                    effect_frame[(effect_hash, effect_fid)] = [(obj_cause_effect_offset_rule[(cause_hash, effect_hash, rule.cause_offset)], obj_cause_effect_offset_times[(cause_hash, effect_hash, rule.cause_offset)])]

        new_new_potential_rules = []
        new_new_seen_rules = []
        #if debug: print('\n-\n\neffect_frame')
        for (effect_hash, fid), conflicting_rules_tuples in effect_frame.items():
            #if debug: print(f'{effect_hash} at {fid} -> {len(conflicting_rules_tuples)} - {[cr[0].my_hash() for cr in conflicting_rules_tuples]}')

            if len(conflicting_rules_tuples) > 1:
                best_times = 0
                best_rule = None
                for conflicting_rule, times in conflicting_rules_tuples:
                    if times > best_times:
                        best_times = times
                        best_rule = conflicting_rule
                if best_rule.my_hash() not in new_new_seen_rules:
                    new_new_potential_rules.append(best_rule)
                    new_new_seen_rules.append(best_rule.my_hash())
                    if best_rule.my_hash() not in seen_rules: seen_rules.append(best_rule.my_hash())
                    obj_explained_score -= best_times
            else:
                if conflicting_rules_tuples[0][0].my_hash() not in new_new_seen_rules:
                    new_new_potential_rules.append(conflicting_rules_tuples[0][0])
                    new_new_seen_rules.append(conflicting_rules_tuples[0][0].my_hash())
                    if conflicting_rules_tuples[0][0].my_hash() not in seen_rules: seen_rules.append(conflicting_rules_tuples[0][0].my_hash())
                    obj_explained_score -= conflicting_rules_tuples[0][1]
        
        explained_score += obj_explained_score

        if debug:
            print('\n-\n')
            for rule_pid, rule in enumerate(new_new_potential_rules): print(f'new_new_potential_rule{rule_pid}: {rule.my_hash()}')

        for rule in new_new_potential_rules:
            obj.add_rule(rule) 
            
        if debug:
            print('\n-\n')
            print(f'obj_explained_score: {obj_explained_score}')
            debug = False
            #exit()

    return explained_score, len(seen_rules)

        