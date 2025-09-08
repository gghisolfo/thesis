from core.individual import Individual
from core.object import Object
from core.prototype import Prototype
from utils.various import ID_generator, compute_diff
from core.unexplained import (
    check_blink, check_disappearance, check_duplication,
    check_for_property0_changes, check_for_speed,
    check_multiple_holes_simple, check_multiple_holes_speed,
    )
from core.rule import new_infer_rules
from core.property import Shape_x, Shape_y

import math
import pickle



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

def create_initial_population_from_prototypes(prototypes, patches_per_frame, global_events_per_frame, resume_population= None, resume_prototype_instances= None, resume_prototype_assignment= None, debug= False):

    if resume_population is None:

        ind_id_generator = ID_generator()
        obj_id_generator = ID_generator()
        proto_id_generator = ID_generator()

        # initialization with one object per patch in the first frame

        present_objects = {obj_id_generator(): Object(frames_id= [0], sequence= [patch], properties= {0: patch.properties}, global_events= {0: global_events_per_frame[0]}) for patch in patches_per_frame[0]} # dict obj_id: obj
        population = {ind_id_generator(): [obj_id for obj_id in present_objects.keys()]} # list of individual, each individual is a list of objects_id
        not_present_objects = {} # dict obj_id: obj

        prototype_instances = {}
        possible_prototype_assignment = {ind_id: {obj_id: None for obj_id in population[ind_id]} for ind_id in population.keys()}

        starting_frame_id = 1

    else:

        starting_frame_id = 0
        max_ind_id = 0
        max_obj_id = 0
        max_proto_id = 0

        population = {}
        all_objects = {}
        possible_prototype_assignment = {}

        for ind_id, ind in resume_population.items():

            ind_obj_ids = []

            if ind_id > max_ind_id: max_ind_id = ind_id

            for obj_id, obj in ind.object_dict.items():

                ind_obj_ids.append(obj_id)

                if obj_id > max_obj_id: max_obj_id = obj_id

                if obj_id not in all_objects.keys(): all_objects[obj_id] = obj

                if obj.frames_id[-1] > starting_frame_id: starting_frame_id = obj.frames_id[-1]

            population[ind_id] = ind_obj_ids
            possible_prototype_assignment[ind_id] = {}
            for obj_id in ind_obj_ids:
                prototype_instances_assigned_to_obj = resume_prototype_assignment[ind_id][obj_id]
                possible_prototype_assignment[ind_id][obj_id] = prototype_instances_assigned_to_obj
                tmp_max_proto_id = max(prototype_instances_assigned_to_obj)
                if tmp_max_proto_id > max_proto_id:
                    max_proto_id = tmp_max_proto_id

            possible_prototype_assignment[ind_id] = {obj_id: resume_prototype_assignment[ind_id][obj_id] for obj_id in ind_obj_ids}

        prototype_instances = {proto_instance_id: proto_info for proto_instance_id, proto_info in resume_prototype_instances}
        # proto_info: (prototype_id, prototype_instance_shape or None if same_shape= False)

        ind_id_generator = ID_generator(max_ind_id)
        obj_id_generator = ID_generator(max_obj_id)
        proto_id_generator = ID_generator(max_proto_id)

        present_objects = {}
        not_present_objects = {}

        for obj_id, obj in all_objects.items():
            if obj.frames_id[-1] == starting_frame_id: present_objects[obj_id] = obj
            else: not_present_objects[obj_id] = obj
        
        starting_frame_id += 1


    survival_dict = {ind_id: SURVIVAL_TIME for ind_id in population.keys()} # ind_id: survival_ages_remaining

    #
    if not debug: print('\n\n\n')
    for frame_id_enum, (patches, global_events) in enumerate(zip(patches_per_frame[starting_frame_id:], global_events_per_frame[starting_frame_id:])):
        frame_id = frame_id_enum + starting_frame_id
        if debug: print(f'\n\n---------------------------------------\nframe: {frame_id}/{len(patches_per_frame)} - population: {len(population.keys())}')
        else: print(f'\033[K\033[F\033[K\033[F\033[K\033[F\033[K\033[F\033[K\n\n---------------------------------------\nframe: {frame_id}/{len(patches_per_frame)} - population: {len(population.keys())}')

        unassigned_objects = [obj_id for obj_id in present_objects.keys()] # all present objects of all individuals (some are shared between individuals)
        unassigned_patches = {ind_id: [p for p in patches] for ind_id in population.keys()} # list of unassigned patches for each individual

        # evaluate perfectly explainable patches (the ones that can be inferred from the current properties (for now: correct no speed, correct speed zero or correct speed))

        if debug: print('\nperfect')

        for pid, patch in enumerate(patches):
            if debug: print(f'\rpatch {pid+1}/{len(patches)}', end= "")
            other_patches = [p for p in patches if p != patch]

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
                    new_properties[frame_id] = prediction
                    current_obj.update(frame_id, patch, new_properties, other_patches, global_events)
                    current_obj.add_explained({frame_id - 1: predicted_explained})
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
            other_patches = [p for p in patches if p != patch]

            assigned_objects = []
            new_unassigned_objects = []

            for obj_id in unassigned_objects:
                current_obj = present_objects[obj_id]

                is_speed, is_confirmed, unexplained_dict, new_properties = check_for_speed(current_obj, patch, frame_id, patches_per_frame[frame_id + 1] if frame_id + 1 < len(patches_per_frame) else None)
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

                        current_obj.update(frame_id, patch, new_properties, other_patches, global_events)
                        current_obj.add_unexplained(unexplained_dict)
                        assigned_objects.append(obj_id)

                        for ind_id in original_inds:
                            unassigned_patches[ind_id].remove(patch)

                    else:

                        new_obj_id = obj_id_generator()
                        new_obj = current_obj.copy()
                        new_obj.update(frame_id, patch, new_properties, other_patches, global_events)
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
                    diff = compute_diff(current_object.properties[frame_id - 1], patch) #here

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

                        p0_did_change, unexplained_dict, new_properties = check_for_property0_changes(present_objects[replacement_obj_id], patch, frame_id)

                        present_objects[replacement_obj_id].update(frame_id, patch, new_properties, [p for p in patches if p != patch], global_events)
                        present_objects[replacement_obj_id].add_unexplained(unexplained_dict)

                        for ind_id in ind_list:
                            population[ind_id] = [ob for ob in population[ind_id] if ob != obj_id] + [replacement_obj_id]
                            remaining_objects[ind_id] = [ob for ob in remaining_objects[ind_id] if ob != obj_id]
                            unassigned_patches[ind_id].remove(patch)

                # update of original object with best_patch and q0 change

                p0_did_change, unexplained_dict, new_properties = check_for_property0_changes(present_objects[obj_id], best_patch, frame_id)

                present_objects[obj_id].update(frame_id, best_patch, new_properties, [p for p in patches if p != best_patch], global_events)
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

                it_disappeared, unexplained_dict, properties = check_disappearance(present_objects[obj_id], frame_id)

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

                    is_simple, unexplained_dict, properties = check_multiple_holes_simple(current_obj, patch, frame_id)
                    if is_simple: unexplained.append((unexplained_dict, properties)) # same speed it has
                    else: # try a new speed (obtained on disappearance)

                        moved_with_new_speed, unexplained_dict, properties = check_multiple_holes_speed(current_obj, patch, frame_id)
                        if moved_with_new_speed: unexplained.append((unexplained_dict, properties))

                    #

                    # blink
                    
                    it_blinked, unexplained_dict, properties = check_blink(current_obj, patch, frame_id)
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
                        new_properties[frame_id] = properties
                        present_objects[new_obj_id] = Object(current_obj.frames_id + [frame_id], current_obj.sequence + [patch], new_properties, new_expl_dict)

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

                is_duplicated, unexplained_dict, properties = check_duplication(current_obj, patch, frame_id)
                if is_duplicated:

                    new_obj_id = obj_id_generator()
                    present_objects[new_obj_id] = Object([frame_id], [patch], {frame_id: properties}, unexplained_dict)

                    #here do it for each ind with obj_id and patch inside

                    new_ind_id = ind_id_generator()
                    new_inds[new_ind_id] = [ob for ob in population[ind_id]] + [new_obj_id]

                #

                # new object

                new_obj_id = obj_id_generator()
                present_objects[new_obj_id] = Object([frame_id], [patch], {frame_id: patch.properties})

                population[ind_id].append(new_obj_id)

        population |= new_inds
        for ind_id in new_inds.keys(): survival_dict[ind_id] = SURVIVAL_TIME

        #

        # rules
        ind_rule_score = {}
        for ind_id, ind in population.items(): ind_rule_score[ind_id] = new_infer_rules(ind, present_objects, not_present_objects, frame_id)
        #print(ind_rule_score)
        #

        # scoring and pruning

        if debug: print(f'\n\nscoring - population: {len(population)}')

        tuple_score = []
        ind_id_score_tuple = []
        for ind_pid, (ind_id, ind) in enumerate(population.items()):
            if debug: print(f'\rind {ind_pid+1}/{len(population)}', end= "")
            total_unexplained = 0

            ind_objects = [(obj_id, present_objects[obj_id] if obj_id in present_objects.keys() else not_present_objects[obj_id]) for obj_id in ind]

            for obj_id, obj in ind_objects:
                for unexpl in obj.unexplained.values():
                    total_unexplained += len(unexpl)
            
            explained_score, rule_score = ind_rule_score[ind_id]
            #tuple_score.append((total_unexplained, explained_score, rule_score))
            #tuple_score.append((explained_score, total_unexplained, rule_score))
            #tuple_score.append((explained_score + total_unexplained, rule_score))
            tuple_score.append((explained_score + total_unexplained + rule_score))
            ind_id_score_tuple.append(ind_id)

        sort_idx = sorted(range(len(tuple_score)), key=lambda i: tuple_score[i])
        tuple_score = [tuple_score[i] for i in sort_idx]
        ind_id_score_tuple = [ind_id_score_tuple[i] for i in sort_idx]

        # removal of pruned individuals

        inds_to_remove = [ind_id_score_tuple[i] for i in range(len(ind_id_score_tuple)) if tuple_score[i] > tuple_score[0]]
        #inds_to_remove = ind_id_score_tuple[1:]
        #inds_to_remove = []

        #probabilistic survival of inds_to_remove ?

        new_inds_to_remove = []
        for ind_id in survival_dict.keys():
            if ind_id in inds_to_remove:
                if survival_dict[ind_id] > 0:
                    survival_dict[ind_id] -= 1
                else: new_inds_to_remove.append(ind_id)
            else: survival_dict[ind_id] = SURVIVAL_TIME
        inds_to_remove = new_inds_to_remove
        remove_inds(population, unassigned_patches, unassigned_objects, present_objects, not_present_objects, survival_dict, inds_to_remove)


    scores_dict = {ind_id_score_tuple[i]: tuple_score[i] for i in range(len(ind_id_score_tuple)) if tuple_score[i] == tuple_score[0]}

    all_obj = present_objects | not_present_objects
    final_population = []
    for ind_id, ind in population.items():
        if ind_id in scores_dict.keys():
            object_dict = {}
            for obj_id, obj in all_obj.items():
                if obj_id in ind:
                    object_dict[obj_id] = obj
            final_population.append((Individual(object_dict, len(patches_per_frame)), scores_dict[ind_id]))

    return final_population


from euristic import summarize_into_prototypes
from utils import load_patches_per_frame

with open('best_individual.pkl', 'rb') as f:
    ind = pickle.load(f)

prototypes = summarize_into_prototypes(ind, debug= True)

exit()

patches_per_frame, global_events_per_frame = load_patches_per_frame(None)
#patches_per_frame, global_events_per_frame = load_patches_per_frame('arkanoid_log_2025_02_07_16_03_00.pkl')

initial_individual, proto_assignment = create_initial_population_from_prototypes(prototypes, patches_per_frame, global_events_per_frame)

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
