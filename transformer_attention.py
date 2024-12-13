# pip install accelerate
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import pprint as pp
import pandas as pd
import os
from tqdm import tqdm


from utilities import get_attention

def get_transformer_attention(
        text_data, # "data/WebQAmGaze/target_experiments_IS_EN.json" 
        model_name="google/gemma-2-2b-it"):
    # Loda data
    all_data = text_data
    # with open(text_data) as f:
    #     all_data = json.load(f)
    print(len(all_data), all_data[0])

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    ## Get transformer attentions by layers
    for item in tqdm(all_data):
        normalized_layer_attention = get_attention(item['text'], 
            item['text'] + ' Answer the question concisely with only the answer: ' + item['question'], 
            tokenizer, model)
        item['normalized_layer_attention'] = normalized_layer_attention
        
    return all_data

## Load human attentions

# for item in all_data:
#     # To load the TRTs and Number of Fixations for each word given a trial one can use:
#     participant_fixation_dictionary = pd.read_csv(os.path.join('data/pre_processed_data/fixation_data_per_part',
#         f"{item['worker_id']}_{item['set_name']}_fix_dict.csv")) 
#     participant_fixation_dictionary= participant_fixation_dictionary[participant_fixation_dictionary['text_id'] == item['text_name']]
#     participant_fixation_dictionary['TRT'] = participant_fixation_dictionary['TRT'].fillna(0)
#     print(participant_fixation_dictionary)
#     print(item['text'])
#     break

## Calaulate

