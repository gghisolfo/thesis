from core.object import Object
from core.property import Shape_x, Shape_y
from euristic import summarize_into_prototypes
from utils import load_patches_per_frame, ID_generator, compute_diff
from core.unexplained import (
    check_blink,
    check_disappearance,
    check_duplication,
    check_for_property0_changes,
    check_for_speed,
    check_multiple_holes_simple,
    check_multiple_holes_speed,
)

import math
import pickle
from itertools import product
from collections import Counter


#####################################
#####################################
#####################################

def assign_prototypes(individual, all_objects, prototypes, ind_id= -1):

    if ind_id == 0: debug= False
    else: debug= False

    # grouping objects by their initial shape

    groups = {}
    for obj_id in individual:
        obj = all_objects[obj_id]
        groups.setdefault((obj.properties[0][Shape_x], obj.properties[0][Shape_y]), []).append(obj_id)

    if debug: print(f'groups: {[f"shape: {shape} -> {[all_objects[obj_id].sequence[0].description for obj_id in objs_id]}" for shape, objs_id in groups.items()]}')
    
    # Build a list of group information.

    group_info_list = []
    for group_idx, (group_shape, obj_ids) in enumerate(groups.items()):
        group_info_list.append({
            'group_id': group_idx,
            'shape': group_shape,
            'objects': obj_ids
        })
    
    # group-prototype matching

    for group in group_info_list:
        possible_assignments = {}  # proto_id -> combined score (sum of individual scores)
        for proto_id, proto in prototypes.items():
            all_possible = True
            total_score = 0
            for obj_id in group['objects']:
                obj = all_objects[obj_id]
                #if proto_id == 1 and 'paddle' in obj.sequence[0].description and ind_id == 1: print('======================================================================== here')
                possible, score = proto.test(obj, ind_id)
                #if ind_id == 1:
                #    print('------')
                #    print(f'prototype {proto_id}')
                #    print(obj.sequence[0].description)
                #    print(f'possible: {possible}')
                if not possible:
                    all_possible = False
                    break
                total_score += score
            if all_possible:
                possible_assignments[proto_id] = total_score
        group['possible_assignments'] = possible_assignments
        # Build a group name: use the first object id plus "_group" if there is more than one object.
        group['group_name'] = f'{group["objects"][0]}' + ("_group" if len(group['objects']) > 1 else "")

    if debug:
        print('\nPhase 2 ended\n\ngroup_info_list: ')
        for group in group_info_list:
            print(group)
    
    # Combination and Instance Assignment.

    group_options = []
    for group in group_info_list:
        opts = [(proto_id, group['possible_assignments'][proto_id]) for proto_id in group['possible_assignments']]
        group_options.append(opts)

    if debug:
        print(f'\noptions:')
        for opts in group_options:
            print(opts)
    
    # Form the full Cartesian product over groups.
    all_combinations = list(product(*group_options))
    
    # For each combination, assign instance ids according to the rules:
    # - For prototypes with same_shape==False: same prototype gets the same instance_id across groups.
    # - For prototypes with same_shape==True: each occurrence gets a new instance_id.
    explanations = []
    for comb in all_combinations:
        comb_explanation = []
        instance_assignment = {}  # For same_shape==False prototypes: proto_id -> instance_id
        instance_counter = 0
        for idx, (proto_id, group_score) in enumerate(comb):
            group = group_info_list[idx]
            proto = prototypes[proto_id]
            if not proto.same_shape:
                # Reuse the same instance id for this prototype if it was already assigned.
                if proto_id in instance_assignment:
                    inst_id = instance_assignment[proto_id]
                else:
                    inst_id = instance_counter
                    instance_assignment[proto_id] = inst_id
                    instance_counter += 1
            else:
                # For same_shape==True, assign a fresh instance id.
                inst_id = instance_counter
                instance_counter += 1
            comb_explanation.append((group['group_name'], group['objects'], proto_id, group_score, inst_id))
        explanations.append(comb_explanation)
    
    return explanations


#####################################
#####################################
#####################################

SURVIVAL_TIME = 2

def remove_inds(population, unassigned_patches, unassigned_objects, present_objects, not_present_objects, survival_dict, inds_to_remove):

    for ind_id in inds_to_remove:
        population.pop(ind_id)
        unassigned_patches.pop(ind_id)
        survival_dict.pop(ind_id)

    # and of objects that are not in kept individuals

    objs_to_keep = []
    for ind in population.values():
        objs_to_keep.extend(ind)
    objs_to_keep = set(objs_to_keep)

    present_to_remove = []
    for obj_id in present_objects.keys():
        if obj_id not in objs_to_keep: present_to_remove.append(obj_id)
    for obj_id in present_to_remove:
        present_objects.pop(obj_id)

    not_present_to_remove = []
    for obj_id in not_present_objects.keys():
        if obj_id not in objs_to_keep: not_present_to_remove.append(obj_id)
    for obj_id in not_present_to_remove:
        not_present_objects.pop(obj_id)

    for obj_id in present_to_remove + not_present_to_remove:
        if obj_id in unassigned_objects: unassigned_objects.remove(obj_id)

def euristic_initialization_with_prototypes(prototypes, patches_frame_0, patches_frame_1, global_events_frame_0, global_events_frame_1, debug= False):

        ind_id_generator = ID_generator()
        obj_id_generator = ID_generator()

        # initialization with one object per patch in the first frame

        present_objects = {obj_id_generator(): Object(frames_id= [0], sequence= [patch], properties= {0: patch.properties}, global_events= {0: global_events_frame_0}) for patch in patches_frame_0} # dict obj_id: obj
        population = {ind_id_generator(): [obj_id for obj_id in present_objects.keys()]} # list of individual, each individual is a list of objects_id
        not_present_objects = {} # dict obj_id: obj
        
        survival_dict = {ind_id: SURVIVAL_TIME for ind_id in population.keys()} # ind_id: survival_ages_remaining

        ############################################
        ############################################
        ############################################

        unassigned_objects = [obj_id for obj_id in present_objects.keys()] # all present objects of all individuals (some are shared between individuals)
        unassigned_patches = {ind_id: [p for p in patches_frame_1] for ind_id in population.keys()} # list of unassigned patches for each individual

        # evaluate perfectly explainable patches (the ones that can be inferred from the current properties (for now: correct no speed, correct speed zero or correct speed))

        if debug: print('\nperfect')

        for pid, patch in enumerate(patches_frame_1):
            if debug: print(f'\rpatch {pid+1}/{len(patches_frame_1)}', end= "")
            other_patches = [p for p in patches_frame_1 if p != patch]

            perfectly_assigned_objects = []
            assigned_patches = {}
            inds_to_remove = []

            for obj_id in unassigned_objects:
                current_obj = present_objects[obj_id]

                prediction = current_obj.prediction
                predicted_explained = current_obj.predicted_explained

                all_ok = True
                for property_class, value in patch.properties.items():

                    if prediction[property_class] != value:
                        all_ok = False
                
                if all_ok: # if an object prediction correctly identifies a patch, then that patch is assigned to the object and marked as assigned for the individuals with that object
                    
                    new_properties = {fid: {k: v for k, v in prop.items()} for fid, prop in current_obj.properties.items()}
                    new_properties[1] = prediction
                    current_obj.update(1, patch, new_properties, other_patches, global_events_frame_1)
                    current_obj.add_explained({0: predicted_explained})
                    perfectly_assigned_objects.append(obj_id)

                    for ind_id, ind in population.items():
                        if obj_id in ind:
                            if ind_id in assigned_patches.keys():
                                if patch not in assigned_patches[ind_id]: assigned_patches[ind_id].append(patch)
                                else:
                                    inds_to_remove.append(ind_id)
                                    continue
                            else: assigned_patches[ind_id] = [patch]
                            unassigned_patches[ind_id].remove(patch)

            remove_inds(population, unassigned_patches, unassigned_objects, present_objects, not_present_objects, survival_dict, inds_to_remove)
            
            for obj_id in perfectly_assigned_objects:
                if obj_id in unassigned_objects: unassigned_objects.remove(obj_id)

        #

        ## evaluate possible solution for Q1 changes (check if a first degree quantity can explain the diff, in that case the change happened in the frame before)

        if debug: print(f'\n\nQ1 - population: {len(population)}')

        pset = set([p for u_p in unassigned_patches.values() for p in u_p]) # set of all the unassigned patches of all individuals
        for pid, patch in enumerate(pset):
            if debug: print(f'\rpatch {pid+1}/{len(pset)}', end= "")
            other_patches = [p for p in patches_frame_1 if p != patch]

            assigned_objects = []
            new_unassigned_objects = []

            for obj_id in unassigned_objects:
                current_obj = present_objects[obj_id]

                is_speed, is_confirmed, unexplained_dict, new_properties = check_for_speed(current_obj, patch, 1, None)
                if is_speed:

                    if is_confirmed:
                        
                        original_inds = []
                        inds_to_branch = []
                        for ind_id, ind in population.items():
                            if obj_id in ind:
                                if patch in unassigned_patches[ind_id]: original_inds.append(ind_id)
                                else: inds_to_branch.append(ind_id)

                        if inds_to_branch:

                            replacement_obj_id = obj_id_generator()
                            replacement_obj = current_obj.copy()
                            present_objects[replacement_obj_id] = replacement_obj
                            new_unassigned_objects.append(replacement_obj_id)

                            for ind_id in inds_to_branch:
                                population[ind_id] = [ob for ob in population[ind_id] if ob != obj_id] + [replacement_obj_id]

                        current_obj.update(1, patch, new_properties, other_patches, global_events_frame_1)
                        current_obj.add_unexplained(unexplained_dict)
                        assigned_objects.append(obj_id)

                        for ind_id in original_inds:
                            unassigned_patches[ind_id].remove(patch)

                    else:

                        new_obj_id = obj_id_generator()
                        new_obj = current_obj.copy()
                        new_obj.update(1, patch, new_properties, other_patches, global_events_frame_1)
                        new_obj.add_unexplained(unexplained_dict)
                        present_objects[new_obj_id] = new_obj

                        new_inds = {}

                        for ind_id, ind in population.items():
                            if obj_id in ind and patch in unassigned_patches[ind_id]:

                                new_ind_id = ind_id_generator()
                                new_inds[new_ind_id] = [ob for ob in population[ind_id] if ob != obj_id] + [new_obj_id]
                                unassigned_patches[new_ind_id] = [p for p in unassigned_patches[ind_id] if p != patch]

                        population |= new_inds
                        for ind_id in new_inds.keys(): survival_dict[ind_id] = SURVIVAL_TIME

            for obj_id in assigned_objects: unassigned_objects.remove(obj_id)
            for obj_id in new_unassigned_objects: unassigned_objects.append(obj_id)

        #

        # evaluate patches and objects with one time quantity change (assignment based on property's proximity, until there are couples)
        
        # volendo qua si potrebbero valutare combinazioni per scomparsa/comparsa, ma forse esploderebbe
        # piu che altro occorrerebbe nel caso magari valutare anche la possibilita che un oggetto prima scomparso stia ricomparendo (oltre che il "nuovo oggetto compare")

        if debug: print(f'\n\nremaining pairing - population: {len(population)}')

        remaining_objects = {ind_id: [ob for ob in unassigned_objects if ob in ind] for ind_id, ind in population.items()}

        op_best_assignments = {} # (obj_id, patch): (times_assigned, list_of_individuals)
        for ind_pid, (ind_id, ind) in enumerate(population.items()):
            
            if debug: print(f'\rind {ind_pid+1}/{len(population)}', end= "")

            op_diff = []

            for obj_id in remaining_objects[ind_id]:
                current_object = present_objects[obj_id]

                for patch in unassigned_patches[ind_id]:
                    diff = compute_diff(current_object.properties[0], patch) #here

                    op_diff.append((obj_id, patch, diff))

            op_diff = sorted(op_diff, key= lambda x: x[2])

            object_with_best_assignment = []
            patch_with_best_assigned = []
            
            for obj_id, patch, diff in op_diff:
                if obj_id not in object_with_best_assignment and patch not in patch_with_best_assigned:

                    object_with_best_assignment.append(obj_id)
                    patch_with_best_assigned.append(patch)

                    if obj_id in op_best_assignments.keys():
                        if patch in op_best_assignments[obj_id].keys():
                            oba = op_best_assignments[obj_id][patch]
                            op_best_assignments[obj_id][patch] = (oba[0] + 1, oba[1] + [ind_id])
                        else:
                            op_best_assignments[obj_id][patch] = (1, [ind_id])
                    else:
                        op_best_assignments[obj_id] = {patch: (1, [ind_id])}

        all_remaining_objects = set()
        for ro in remaining_objects.values(): all_remaining_objects.update(ro)

        for obj_id in list(all_remaining_objects):
            if obj_id in op_best_assignments.keys():
                best_patch = None
                best_times = 0
                best_ind_list = None

                for patch, (times, ind_list) in op_best_assignments[obj_id].items():
                    if times > best_times:
                        best_times = times
                        best_patch = patch
                        best_ind_list = ind_list

                for patch, (times, ind_list) in op_best_assignments[obj_id].items():
                    if patch != best_patch:

                        # branching new object with that patch and q0 change, for all ind in ind_list

                        replacement_obj_id = obj_id_generator()
                        present_objects[replacement_obj_id] = present_objects[obj_id].copy()

                        p0_did_change, unexplained_dict, new_properties = check_for_property0_changes(present_objects[replacement_obj_id], patch, 1)

                        present_objects[replacement_obj_id].update(1, patch, new_properties, [p for p in patches_frame_1 if p != patch], global_events_frame_1)
                        present_objects[replacement_obj_id].add_unexplained(unexplained_dict)

                        for ind_id in ind_list:
                            population[ind_id] = [ob for ob in population[ind_id] if ob != obj_id] + [replacement_obj_id]
                            remaining_objects[ind_id] = [ob for ob in remaining_objects[ind_id] if ob != obj_id]
                            unassigned_patches[ind_id].remove(patch)

                # update of original object with best_patch and q0 change

                p0_did_change, unexplained_dict, new_properties = check_for_property0_changes(present_objects[obj_id], best_patch, 1)

                present_objects[obj_id].update(1, best_patch, new_properties, [p for p in patches_frame_1 if p != best_patch], global_events_frame_1)
                present_objects[obj_id].add_unexplained(unexplained_dict)

                for ind_id in best_ind_list:
                    remaining_objects[ind_id].remove(obj_id)
                    unassigned_patches[ind_id].remove(best_patch)


        # evaluate remaining patches or objects (only one type of the two should remain) (if there are patches left they are new object appeared or previously disappeared objects reappeared, else if there are object left they disappear)

        if debug: print(f'\n\nremaining single - population: {len(population)}')

        new_inds = {}

        for ind_id, ind in population.items(): assert(not (remaining_objects[ind_id] and unassigned_patches[ind_id]))

        for ind_pid, (ind_id, ind) in enumerate(population.items()):
            
            if debug: print(f'\rind {ind_pid+1}/{len(population)}', end= "")

            # disappearing objects

            disappeared = []

            for obj_id in remaining_objects[ind_id]:

                it_disappeared, unexplained_dict, properties = check_disappearance(present_objects[obj_id], 1)

                if it_disappeared:

                    disappearing_obj = present_objects.pop(obj_id)
                    disappearing_obj.add_unexplained(unexplained_dict)
                    not_present_objects[obj_id] = disappearing_obj

                    disappeared.append(obj_id)
                    
            for obj_id in disappeared:
                for ro in remaining_objects.values():
                    if obj_id in ro: ro.remove(obj_id)

            #

            # appearing objects (not yet tested)

            for patch in unassigned_patches[ind_id]:

                # reappearing objects

                assigned_to = None
                best_diff = math.inf

                for obj_id in not_present_objects.keys():
                    if obj_id in ind:

                        diff = compute_diff(not_present_objects[obj_id].prediction, patch)

                        if diff < best_diff:
                            best_diff = diff
                            assigned_to = obj_id

                # magari provare i primi n più simili (2?)

                if assigned_to: # search for a solution that explains the object reappearing

                    current_obj = not_present_objects[assigned_to]

                    unexplained = []

                    #

                    # speed while disappeared

                    is_simple, unexplained_dict, properties = check_multiple_holes_simple(current_obj, patch, 1)
                    if is_simple: unexplained.append((unexplained_dict, properties)) # same speed it has
                    else: # try a new speed (obtained on disappearance)

                        moved_with_new_speed, unexplained_dict, properties = check_multiple_holes_speed(current_obj, patch, 1)
                        if moved_with_new_speed: unexplained.append((unexplained_dict, properties))

                    #

                    # blink
                    
                    it_blinked, unexplained_dict, properties = check_blink(current_obj, patch, 1)
                    if it_blinked: # teleportation mantaining properties (properties changed saved in unexplained)
                        unexplained.append((unexplained_dict, properties))

                    #

                    for unexplained_dict, properties in unexplained:

                        new_obj_id = obj_id_generator()
                        new_expl_dict = {k: [ex for ex in v] for k, v in current_obj.unexplained.items()}
                        for k, v in unexplained_dict.items():
                            if k in new_expl_dict.keys(): new_expl_dict[k].extend(v)
                            else: new_expl_dict[k] = v
                        new_properties = current_obj.properties
                        new_properties[1] = properties
                        present_objects[new_obj_id] = Object(current_obj.frames_id + [1], current_obj.sequence + [patch], new_properties, new_expl_dict)

                        #here do it for each ind with obj_id and patch inside

                        new_ind_id = ind_id_generator()
                        new_inds[new_ind_id] = [ob for ob in population[ind_id] if ob != obj_id] + [new_obj_id]

                #

                # duplication of one object

                assigned_to = None
                best_diff = math.inf

                for obj_id in present_objects.keys():
                    if obj_id in ind:

                        diff = compute_diff(present_objects[obj_id].prediction, patch)

                        if diff < best_diff:
                            best_diff = diff
                            assigned_to = obj_id

                current_obj = present_objects[assigned_to]

                is_duplicated, unexplained_dict, properties = check_duplication(current_obj, patch, 1)
                if is_duplicated:

                    new_obj_id = obj_id_generator()
                    present_objects[new_obj_id] = Object([1], [patch], {1: properties}, unexplained_dict)

                    #here do it for each ind with obj_id and patch inside

                    new_ind_id = ind_id_generator()
                    new_inds[new_ind_id] = [ob for ob in population[ind_id]] + [new_obj_id]

                #

                # new object

                new_obj_id = obj_id_generator()
                present_objects[new_obj_id] = Object([1], [patch], {1: patch.properties})

                population[ind_id].append(new_obj_id)

        population |= new_inds
        for ind_id in new_inds.keys(): survival_dict[ind_id] = SURVIVAL_TIME

        ############################################
        ############################################
        ############################################

#        no_assignment_objects = []
#        possible_assignments = {}
#        confirmed_assignments = {}
#        for obj_id, obj in (present_objects | not_present_objects).items():
#            possible_protos = []
#            for proto_id, proto in prototypes.items():
#                possible, proto_score = proto.test(obj)
#                if possible:
#                    possible_protos.append((proto_id, proto_score))
#
#            if not possible_protos:
#                no_assignment_objects.append(obj_id)
#                continue
#
#            best_proto_score = max(pp[1] for pp in possible_protos)
#            best_possible_protos = [pp[0] for pp in possible_protos if pp[1] == best_proto_score]
#
#            if len(best_possible_protos) == 1:
#                confirmed_assignments[obj_id] = best_possible_protos[0]
#            else:
#                possible_assignments[obj_id] = best_possible_protos
#
#        inds_to_remove = set()
#        for obj_id in no_assignment_objects:
#            for ind_id, ind in population.items():
#                if obj_id in ind: inds_to_remove.update(ind_id)
#        inds_to_remove = list(inds_to_remove)

        all_objects = (present_objects | not_present_objects)

        wrong_individuals = []
        correct_individuals = {}
        
        for ind_id, ind in population.items():
            explanation = assign_prototypes(ind, all_objects, prototypes, ind_id)

            if len(explanation) == 0:
                wrong_individuals.append(ind_id)
                continue

            mapping_scores = []

            for expl_id, mapping in enumerate(explanation):

                mapping_score = 0
                instances_id = set()
                proto_ids_used = set()
                obj_id_used = set()

                for name, objs_id, proto_id, score, instance_id in mapping:

                    mapping_score += score
                    instances_id.update([instance_id])
                    proto_ids_used.update([proto_id])
                    obj_id_used.update(objs_id)
                
                mapping_score += len(obj_id_used) + len(proto_ids_used) - len(instances_id)

                mapping_scores.append((mapping, mapping_score))

            best_mapping_score = max(ms[1] for ms in mapping_scores)
            mapping_scores = [mp for mp in mapping_scores if mp[1] >= best_mapping_score]

            correct_individuals[ind_id] = (len(explanation), mapping_scores)
                    
        #exit()

#        print('\n\nprinting population differences\n')
#
#        all_common_ids = None
#        for ind in population.values():
#            if all_common_ids is None:
#                all_common_ids = set(ind)
#            else:
#                all_common_ids &= set(ind)
#
#        all_objects = present_objects | not_present_objects
#
#        for ind_id, ind in population.items():
#            print(f'\n=============================\ndifferences for ind_{ind_id}:')
#            characterizing_objects = set(ind).difference(all_common_ids)
#            for obj_id in characterizing_objects:
#                print(f'\tobj_{obj_id}: {all_objects[obj_id]}')
#
#            if ind_id in wrong_individuals: print('\n\nTHE INDIVIDUAL HAS NO EXPLANATION GIVEN THE PROTOTYPES\n\n')
#            else:
#                print(f'\nn° total possible mappings: {correct_individuals[ind_id][0]}')
#                print(f'\nn° best possible mappings: {len(correct_individuals[ind_id][1])}')
#                print('\nbest possible mappings:')
#
#                for expl_id, (mapping, mapping_score) in enumerate(correct_individuals[ind_id][1]):
#
#                    print('---------------')
#                    print(f'explanation {expl_id}')
#
#                    for name, objs_id, proto_id, score, instance_id in mapping:
#                        print(f'\t{name} ({objs_id}) assigned to instance {instance_id} (prototype {proto_id}) with score {score}')
#
#                    print(f'\nmapping score: {mapping_score}\n')

        remove_inds(population, unassigned_patches, unassigned_objects, present_objects, not_present_objects, survival_dict, wrong_individuals)

        mappings = {}
        for ind_id in population.keys():
            mappings[ind_id] = [mt for mt in correct_individuals[ind_id][1]]

        if debug:
            print('=======================================================================================')
            print('=======================================================================================')
            print('=======================================================================================')
            print('=======================================================================================')
            print('=======================================================================================')

            print('\nPOPULATION:')
            
            for ind_id, ind in population.items():
                print(f'\n=============================\nind_{ind_id}\n=============================\n')
                for obj_id in ind:
                    print(f'\tobj_{obj_id}: {[p.description for p in all_objects[obj_id].sequence]}')

                print(f'\nn° best possible mappings: {len(mappings[ind_id])}')
                print('\nPossible Mappings:')

                for expl_id, (mapping, mapping_score) in enumerate(mappings[ind_id]):

                    print('---------------')
                    print(f'Mapping {expl_id}')

                    for name, objs_id, proto_id, score, instance_id in mapping:
                        print(f'\t{name} ({objs_id}) assigned to instance {instance_id} (prototype {proto_id}) with score {score}')

                    print(f'\nMapping Score: {mapping_score}\n')

        return population, present_objects, not_present_objects, mappings

def euristic_frame_by_frame_with_prototypes(prototypes, population, present_objects, not_present_objects, mappings, new_patches, new_global_events, debug= False):

    pass

def predict_from_mappings(prototypes, population, present_objects, not_present_objects, mappings, debug= False):

    all_objects = present_objects | not_present_objects

    predictions = {}

    for ind_id, ind in population.items():
        predictions[ind_id] = {}
        for obj_id in ind:
            obj_predictions = []
            for mapping, _ in mappings[ind_id]:
                for _, objs_id, proto_id, _, _ in mapping:
                    if obj_id in objs_id:

                        dummy_object = all_objects[obj_id].copy()

                        for rule in prototypes[proto_id].rules:
                            dummy_object.add_rule(rule)

                        prediction, _ = dummy_object.compute_next(dummy_object.frames_id[-1])
                        obj_predictions.append(prediction)

            if all(op == obj_predictions[0] for op in obj_predictions):

                op_string = ''
                for prop_name, value in sorted([(prop_class.name(), value) for prop_class, value in obj_predictions[0].items()], key= lambda x: x[0]):
                    op_string += f'{prop_name}{value}'

                predictions[ind_id][obj_id] = (op_string, obj_predictions[0])
            
            else: # qui è un po greedy

                hashable_preds_dict = {}
                hashable_preds = []
                for op in obj_predictions:
                    op_string = ''
                    for prop_name, value in sorted([(prop_class.name(), value) for prop_class, value in op.items()], key= lambda x: x[0]):
                        op_string += f'{prop_name}{value}'
                    hashable_preds.append(op_string)
                    hashable_preds_dict[op_string] = op

                most_common_string, _ = Counter(hashable_preds).most_common(1)[0]

                predictions[ind_id][obj_id] = (most_common_string, hashable_preds_dict[most_common_string])

    hashable_inds_dict = {}
    hashable_inds = []
    for ind_id, objs_prediction in predictions.items():
        ind_string = ''
        for obj_string, _ in sorted(objs_prediction.values(), key= lambda x: x[0]):
            ind_string += obj_string
        hashable_inds.append(ind_string)
        hashable_inds_dict[ind_string] = objs_prediction
    
    most_common_string, _ = Counter(hashable_inds).most_common(1)[0]

    prediction = {obj_id: obj_prediction for obj_id, (_, obj_prediction) in enumerate(hashable_inds_dict[most_common_string].values())}

    return prediction


with open('best_individual.pkl', 'rb') as f:
    ind = pickle.load(f)

prototypes = summarize_into_prototypes(ind, debug= True)

patches_per_frame, global_events_per_frame = load_patches_per_frame(None)
#patches_per_frame, global_events_per_frame = load_patches_per_frame('arkanoid_log_2025_02_07_16_03_00.pkl')

initial_population, initial_present_objects, initial_not_present_objects, initial_mappings = euristic_initialization_with_prototypes(prototypes, patches_per_frame[0], patches_per_frame[1], global_events_per_frame[0], global_events_per_frame[1], debug= True)

chosen_ind_id, ind = initial_population.items()[0]

prediction = predict_from_mappings(prototypes, initial_population, initial_present_objects, initial_not_present_objects, initial_mappings)

# fit previous population -> prediction per obj_ids (e quindi order da passare al RL)

print('\n========================\nPrediction from initial_population:\n')
for obj_id, properties in prediction.items():
    print(f'\n----------------\nobj_{obj_id}:\n')
    for prop_class, value in properties.items():
        print(f'\t{prop_class.name()}: {value}')

#out_string = ''
#ind = initial_individual
#for obj_id in ind.object_dict.keys():
#    out_string += f'\n\nobj_{obj_id}'
#    if obj_id in ind.rules.keys():
#        out_string += f'\n\nrules: {ind.rules[obj_id]}\n'
#    else:
#        out_string += '\n\nno rules\n'
#    for frame_id, frame_dict in ind.object_info[obj_id].items():
#        if frame_dict['present']:
#            out_string += f'\nframe {frame_id} - patch: {frame_dict["patch"]}\n- unexplained: {frame_dict["unexplained"]}\n- explained: {frame_dict["explained_unexplained"]}\n- events: {frame_dict["events"]}\n- global events: {frame_dict["global_events"]}'
#        else:
#            out_string += f'\nframe {frame_id} - patch not present\n- unexplained: {frame_dict["unexplained"]}\n- explained: {frame_dict["explained_unexplained"]}\n- events: {frame_dict["events"]}\n- global events: {frame_dict["global_events"]}'
#    pids = sorted([(proto_ids[pkey], pkey) for pkey in proto_assignment[obj_id]], key= lambda x: x[0])
#    out_string += f'\n\npossible prototypes: {[pp[0] for pp in pids]}:'
#    for _, pkey in pids: out_string += f'\n\t{proto_ids[pkey]}: {prototypes[pkey]}'
#
#with open('log_try.txt', 'w') as f:
#    f.write(out_string)
