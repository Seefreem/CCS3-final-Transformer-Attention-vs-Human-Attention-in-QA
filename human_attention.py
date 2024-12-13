import pandas as pd
import numpy as np
import json
import os
import pprint as pp

def normalize(matrix):
    # Only this is changed to use 2-norm put 2 instead of 1
    norm = np.linalg.norm(matrix, 1)
    if norm > 0.0:
        # normalized matrix
        matrix = matrix/norm
    
    return norm, matrix

def get_human_attentions(data_list):

    # Load files
    FIXATION_DATA_FOLDER = os.path.join("./data/WebQAmGaze/pre_processed_data","fixation_data_per_part")
    # PRE_PROCESS_DATA_FOLDER = "./data/WebQAmGaze"
    # experiments_config_file = 'target_experiments_IS_EN.json'
    # data_list = []
    # with open(os.path.join(PRE_PROCESS_DATA_FOLDER, experiments_config_file)) as f:
    #     data_list = json.load(f)
    # len(data_list), data_list[0]
    # Load TRT and FIxations of each sentence of each experiment

    # To load the TRTs and Number of Fixations for each word given a trial one can use:
    new_data_list = []
    for entry in data_list:
        participant_fixation_dictionary = pd.read_csv(
            os.path.join(FIXATION_DATA_FOLDER,
                f"{entry['worker_id']}_{entry['set_name']}_fix_dict.csv")) 
        # print(participant_fixation_dictionary)
        # To see the TRTs and FixationCounts for words in the text:
        # print(entry['text_name'])
        TRTs_and_FixationCounts_sen = participant_fixation_dictionary[participant_fixation_dictionary['text_id'] == entry['text_name']]
        # Repalce NaN with zero on all columns 
        TRTs_and_FixationCounts_sen = TRTs_and_FixationCounts_sen.fillna(0)
        trt_scores = TRTs_and_FixationCounts_sen['TRT'].to_numpy()
        fixation_scores = TRTs_and_FixationCounts_sen['FixCount'].to_numpy()
        # truncate attention vectors
        if 'normalized_layer_attention' in entry.keys():
            (layer, dimension) = entry['normalized_layer_attention'].shape
            min_length = min(min(dimension, len(trt_scores)), len(fixation_scores))
            trt_scores = np.copy(trt_scores[:min_length])
            fixation_scores = np.copy(fixation_scores[:min_length])
            entry['normalized_layer_attention'] = np.copy(entry['normalized_layer_attention'][:, :min_length])
            # print("Length: ", dimension, len(trt_scores), len(fixation_scores), min_length)
        norm_1, matrix = normalize(trt_scores)
        entry['trt'] = matrix
        # print(np.sum(matrix))
        norm_2, matrix = normalize(fixation_scores)
        entry['fixation'] = matrix
        # print(np.sum(matrix))
        if norm_1 > 0.0 and norm_2 > 0.0:
            new_data_list.append(entry)
    data_list = new_data_list    
    data_list[0]

    # Group by participants
    from collections import defaultdict
    data_by_participants = defaultdict(list)
    for data in data_list:
        data_by_participants[data['worker_id']].append(data)
    len(data_by_participants.keys()) # TODO check here. The number of workers does not match.
    print(f"In total, there are {len(data_by_participants.keys())} participants.")

    # Group by sentences
    data_by_sentences = defaultdict(list)
    for data in data_list:
        data_by_sentences[data['text_name']].append(data)
    len(data_by_sentences.keys()) # TODO check here. The number of workers does not match.
    print(f"In total, there are {len(data_by_sentences.keys())} texts.")
    
    # the human relative importance by sentences
    for key in data_by_sentences:
        data = data_by_sentences[key]
        size = len(data)
        if size != 0:
            importance_vector_trt = data[0]['trt']
            importance_vector_fixation = data[0]['fixation']
            # print(importance_vector_fixation[:5])
            for i in range(1, size):
                importance_vector_trt += data[i]['trt']
                importance_vector_fixation += data[i]['fixation']
                # print(importance_vector_fixation[:5]) 
            importance_vector_trt /= size
            importance_vector_fixation /= size
            # print(importance_vector_fixation[:5])
            
            print('np.sum(importance_vector_trt): ', np.sum(importance_vector_trt))
            print('np.sum(importance_vector_fixation):', np.sum(importance_vector_fixation))

    return data_list, data_by_participants, data_by_sentences