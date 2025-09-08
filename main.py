from utils import load_patches_per_frame
from euristic import euristic_initialization, summarize_into_prototypes
from utils import debug_patches_per_frame

import pickle

def main():

    simple_arkanoid_log_file_name = 'arkanoid_log_2025_02_07_16_03_00.pkl'
    complete_arkanoid_lose_log_file_name = 'arkanoid_log_2025_02_07_16_06_40.pkl'
    complete_arkanoid_win_log_file_name = 'arkanoid_log_2025_02_07_10_58_26.pkl'
    complete_arkanoid_win_2_log_file_name = 'arkanoid_log_2025_02_07_15_44_51.pkl'
    
    patches_per_frame, global_events_per_frame = load_patches_per_frame(None) # last log

    #patches_per_frame = debug_patches_per_frame
    #global_events_per_frame = [[] for _ in range(len(debug_patches_per_frame))]

    #patches_per_frame, global_events_per_frame = load_patches_per_frame(simple_arkanoid_log_file_name)

    #patches_per_frame, global_events_per_frame = load_patches_per_frame(complete_arkanoid_lose_log_file_name)

    #patches_per_frame, global_events_per_frame = load_patches_per_frame(complete_arkanoid_win_log_file_name)

    #from core.property import Pos_x, Pos_y
    #for i in range(20):
    #    print(f'---------------\nframe_{i}')
    #    print(f'global_events: {global_events_per_frame[i]}')
    #    for patch in patches_per_frame[i]:
    #        if patch.description == 'paddle_center': print(f'pos: ({patch.properties[Pos_x]}, {patch.properties[Pos_y]})')
    #exit()

    #### ------------------------------- ####

    #population = euristic_initialization(patches_per_frame, global_events_per_frame)
    stop_at = -15
    population = euristic_initialization(patches_per_frame[:stop_at], global_events_per_frame[:stop_at])
    
    print('\n\n=====================================\neuristic_initialization end\n=====================================\n')

    scores = []
    for ind_id, (ind, score) in enumerate(population):
        scores.append((ind_id, ind, score))
    
    population = [(ind_id, ind, score) for ind_id, ind, score in sorted(scores, key= lambda x: x[2])]#[:1]

    out_string = ''
    for ind_id, ind, score in population:
        out_string += f'\n--------------\nind_{ind_id}:\n--------------'
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
        out_string += f'\nscore: {score}'
    with open('log.txt', 'w') as f:
        f.write(out_string)
    print(f'n° individuals: {len(population)}')
    print(f'best score: {population[0][2]}')

    with open('best_population.pkl', 'wb') as f:
        pickle.dump({tup[0]: tup[1] for tup in population}, f)
    
    with open('best_individual.pkl', 'wb') as f:
        pickle.dump(population[0][1], f)

    #exit()

    summarize_into_prototypes(population[0][1])

    with open('best_population.pkl', 'rb') as f:
        population = euristic_initialization(patches_per_frame, global_events_per_frame, resume_population= pickle.load(f))
    
    print('\n\n=====================================\neuristic_initialization resume end\n=====================================\n')

    scores = []
    for ind_id, (ind, score) in enumerate(population):
        scores.append((ind_id, ind, score))
    
    population = [(ind_id, ind, score) for ind_id, ind, score in sorted(scores, key= lambda x: x[2])]#[:1]

    out_string = ''
    for ind_id, ind, score in population:
        out_string += f'\n--------------\nind_{ind_id}:\n--------------'
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
        out_string += f'\nscore: {score}'
    with open('log_resume.txt', 'w') as f:
        f.write(out_string)
    print(f'n° individuals: {len(population)}')

    with open('best_population_resume.pkl', 'wb') as f:
        pickle.dump({tup[0]: tup[1] for tup in population}, f)
    
    with open('best_individual_resume.pkl', 'wb') as f:
        pickle.dump(population[0][1], f)

    summarize_into_prototypes(population[0][1])

    return

if __name__ == "__main__":
    main()
