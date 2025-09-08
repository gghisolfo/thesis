import pickle
import copy
import numpy as np
from scipy.optimize import linear_sum_assignment
import itertools

from core.object import Object
from core.individual import Individual
from utils.various import compute_diff
from utils import load_patches_per_frame
from euristic import summarize_into_prototypes
from core.property import Shape_x, Shape_y

# ---------------------------------------------------------------------------
def patch_pair_matches_proto_signature(proto, patch0, patch1):
    """
    Checks whether the candidate patch pair (one from frame 0 and one from frame 1)
    satisfies the constraints expressed in the prototype’s prop_behavior_signature.
    
    Here the signature is assumed to be a tuple of (prop_class, behavior) pairs.
    For every property marked as "const", the value in patch0 must equal the value in patch1.
    (For "var" properties, no equality is imposed.)
    """
    for (prop_class, behavior) in proto.get('prop_behavior_signature', []):
        value0 = patch0.properties.get(prop_class)
        value1 = patch1.properties.get(prop_class)
        if behavior == 'const':
            if value0 is None or value1 is None or value0 != value1:
                return False
    return True

# ---------------------------------------------------------------------------
def confirm_rule(rule, obj, patch1):
    """
    Checks whether a given rule is confirmed on the candidate object.
    It assumes the cause frame is the object's current frame (here, frame 0 in our candidate objects).
    For each effect (with effect.property_class), it computes:
         predicted_value = effect.a * (previous value) + effect.b
    and compares it with the actual value in patch1.properties.
    """
    cause_frame = obj.frames_id[-1]
    try:
        triggered, effects, offset = rule.trigger(obj, cause_frame)
    except Exception:
        return False
    if not triggered:
        return False
    for effect in effects:
        prop = effect.property_class
        previous_value = obj.properties[cause_frame].get(prop)
        if previous_value is None:
            return False
        predicted_value = effect.a * previous_value + effect.b
        actual_value = patch1.properties.get(prop)
        if actual_value is None or actual_value != predicted_value:
            return False
    return True

# ---------------------------------------------------------------------------
def create_initial_population_from_prototypes(prototypes, patches_per_frame, global_events_per_frame):
    """
    Creates an initial population from the first two frames using an optimal matching (Hungarian algorithm).
    
    For each candidate patch pair (one patch from frame 0 and one from frame 1) that satisfies at least one prototype’s
    prop_behavior_signature, we compute a cost (using compute_diff on the patch property dictionaries);
    otherwise, we assign a high cost.
    
    The Hungarian algorithm is then used to produce a one-to-one assignment.
    
    For every valid assignment, we create a candidate object from patch0, update it with patch1 (which computes
    the object's prediction), and then compute a score for each candidate prototype.
    
    The score for each candidate prototype is defined as:
      score = (# of "var" properties for which the value in frame 0 differs from that in patch1) +
              (# of rules in that prototype (from the 'rules' key) that are confirmed via confirm_rule)
    
    For each candidate object we record:
      - The candidate prototype set (a set of prototype IDs),
      - A representative shape (obtained via get_shape_from_properties(patch1.properties)),
      - And a score dictionary (mapping each candidate prototype ID to its score).
    
    The function returns a tuple (population, proto_assignment), where population is a list containing one
    Individual (with its object_dict holding the candidate objects), and proto_assignment is a dict mapping
    candidate object indices to (candidate_proto_set, rep_shape, score_dict).
    """
    frame0 = patches_per_frame[0]
    frame1 = patches_per_frame[1]
    events0 = global_events_per_frame[0]
    events1 = global_events_per_frame[1]
    
    n = len(frame0)
    m = len(frame1)
    cost_matrix = np.zeros((n, m))
    candidate_proto_set = {}  # key: (i, j) -> set of prototype IDs
    
    high_cost = 1e6  # a very high cost for invalid pairs
    
    # Build cost matrix and candidate_proto_set based solely on the "const" constraints.
    for i, patch0 in enumerate(frame0):
        for j, patch1 in enumerate(frame1):
            proto_set = set()
            for proto_id, proto in prototypes.items():
                if patch_pair_matches_proto_signature(proto, patch0, patch1):
                    proto_set.add(proto_id)
            if proto_set:
                cost = compute_diff(patch0.properties, patch1)
                cost_matrix[i, j] = cost
                candidate_proto_set[(i, j)] = proto_set
            else:
                cost_matrix[i, j] = high_cost
    
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    candidate_objects = []
    proto_assignment = {}  # Mapping: candidate object index -> candidate_proto_set
    
    for i, j in zip(row_ind, col_ind):
        if cost_matrix[i, j] >= high_cost:
            continue
        patch0 = frame0[i]
        patch1 = frame1[j]
        proto_set = candidate_proto_set.get((i, j), set())
        
        # Create a candidate object from patch0.
        obj = Object(
            frames_id=[0],
            sequence=[patch0],
            properties={0: copy.deepcopy(patch0.properties)},
            global_events={0: events0}
        )
        # Record candidate prototype set (as a set of prototype IDs) in the object.
        obj.prototype_info = {'candidate_protos': proto_set}
        # Get representative shape from patch1.
    
        
        # Update the object with patch1 (this computes obj.prediction).
        new_properties = copy.deepcopy(obj.properties)
        new_properties[1] = copy.deepcopy(patch1.properties)
        other_patches = [p for p in frame1 if p is not patch1]
        obj.update(
            frame_id=1,
            patch=patch1,
            new_properties=new_properties,
            other_patches=other_patches,
            global_events=events1
        )
        
        candidate_objects.append(obj)
        proto_assignment[len(candidate_objects)-1] = proto_set
    
    obj_dict = {i: obj for i, obj in enumerate(candidate_objects)}
    individual = Individual(obj_dict, last_frame_id=2)
    return individual, proto_assignment




with open('best_individual.pkl', 'rb') as f:
    ind = pickle.load(f)

prototypes = summarize_into_prototypes(ind)

proto_ids = {}
for pkey in prototypes.keys():
    proto_ids[pkey] = len(proto_ids)

patches_per_frame, global_events_per_frame = load_patches_per_frame(None)
#patches_per_frame, global_events_per_frame = load_patches_per_frame('arkanoid_log_2025_02_07_16_03_00.pkl')

initial_individual, proto_assignment = create_initial_population_from_prototypes(prototypes, patches_per_frame, global_events_per_frame)

out_string = ''
ind = initial_individual
for obj_id in ind.object_dict.keys():
    out_string += f'\n\nobj_{obj_id}'
    if obj_id in ind.rules.keys():
        out_string += f'\n\nrules: {ind.rules[obj_id]}\n'
    else:
        out_string += '\n\nno rules\n'
    for frame_id, frame_dict in ind.object_info[obj_id].items():
        if frame_dict['present']:
            out_string += f'\nframe {frame_id} - patch: {frame_dict["patch"]}\n- unexplained: {frame_dict["unexplained"]}\n- explained: {frame_dict["explained_unexplained"]}\n- events: {frame_dict["events"]}\n- global events: {frame_dict["global_events"]}'
        else:
            out_string += f'\nframe {frame_id} - patch not present\n- unexplained: {frame_dict["unexplained"]}\n- explained: {frame_dict["explained_unexplained"]}\n- events: {frame_dict["events"]}\n- global events: {frame_dict["global_events"]}'
    pids = sorted([(proto_ids[pkey], pkey) for pkey in proto_assignment[obj_id]], key= lambda x: x[0])
    out_string += f'\n\npossible prototypes: {[pp[0] for pp in pids]}:'
    for _, pkey in pids: out_string += f'\n\t{proto_ids[pkey]}: {prototypes[pkey]}'

with open('log_try.txt', 'w') as f:
    f.write(out_string)
