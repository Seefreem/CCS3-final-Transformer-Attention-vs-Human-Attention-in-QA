import json
import argparse
import torch

from transformer_attention import get_transformer_attention
from human_attention import get_human_attentions
from utilities import layer_level_correlation, entry_level_correlation, text_level_correlation, participant_level_correlation



def main(data_file, model):
    all_data = []
    with open(data_file) as f:
        all_data = json.load(f)
    # test phrase
    # all_data = all_data[:10]
    all_data = get_transformer_attention(all_data, model)
    # Be aware that the length of a human attention vector may very from the length of taht of transformer's
    # The error was introduced by OCR, so we simply truncate the longer vector
    data_list, data_by_participants, data_by_sentences = get_human_attentions(all_data)
    print("An example of data item with both attentions from a human and a Transformer model:\n", data_list[0])

    ## Calculate correlation scores
    # layer level
    print("Model name:", model)
    data_list_with_correlation_scores, best_correlated_layer_index, layer_correlation = layer_level_correlation(data_list)
    with open('attention_scores.json', 'w') as f:
        json.dump(data_list, f, indent=4)
    print('The best correlated layer index is:', best_correlated_layer_index)
    print('Layer level correlation table:\n', layer_correlation)
    layer_correlation_filename = "layer_correlation.json"
    print('Layer level correlation scores are saved into file', layer_correlation_filename)
    with open(layer_correlation_filename, "w") as outfile:
        json.dump(layer_correlation, outfile, indent=4)

    # entry level
    entry_correlation = entry_level_correlation(data_list_with_correlation_scores, best_correlated_layer_index)
    entry_correlation_filename = "entry_correlation.json"
    print('Entry level correlation scores are saved into file', entry_correlation_filename)
    with open(entry_correlation_filename, "w") as outfile:
        json.dump(entry_correlation, outfile, indent=4)

    # text level
    text_correlation = text_level_correlation(data_list_with_correlation_scores, data_by_sentences, best_correlated_layer_index)
    text_correlation_filename = "text_correlation.json"
    print('Text level correlation scores are saved into file', text_correlation_filename)
    with open(text_correlation_filename, "w") as outfile:
        json.dump(text_correlation, outfile, indent=4)

    # participant level  
    participant_correlation = participant_level_correlation(data_list_with_correlation_scores, data_by_participants, best_correlated_layer_index)
    participant_correlation_filename = "participant_correlation.json"
    print('Participant level correlation scores are saved into file', participant_correlation_filename)
    with open(participant_correlation_filename, "w") as outfile:
        json.dump(participant_correlation, outfile, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()
    main(args.data_file, args.model)

# python correlation.py --data_file data/WebQAmGaze/target_experiments_IS_EN.json --model google/gemma-2-2b-it