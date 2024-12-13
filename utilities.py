import torch
from scipy import stats
import numpy as np
from scipy.special import rel_entr

def layer_level_correlation(data_list):
    
    no_layer = data_list[0]['normalized_layer_attention'].shape[0]
    print('no_layer:', no_layer, data_list[0]['normalized_layer_attention'].shape)
    layer_trt_correlations_spearman = [[] for i in range(no_layer)]
    layer_fixation_correlations_spearman = [[] for i in range(no_layer)]
    layer_trt_p_value_spearman = [[] for i in range(no_layer)]
    layer_fixation_p_value_spearman = [[] for i in range(no_layer)]
    layer_trt_correlations_kl = [[] for i in range(no_layer)]
    layer_fixation_correlations_kl = [[] for i in range(no_layer)]
    
    for data in data_list:
        data['layer_trt_correlations_spearman'] = []
        data['layer_fixation_correlations_spearman'] = []
        data['layer_trt_correlations_kl'] = []
        data['layer_fixation_correlations_kl'] = []
        
        for layer_no in range(no_layer):
            trt_res = stats.spearmanr(data['normalized_layer_attention'][layer_no, :], data['trt'])
            fixation_res = stats.spearmanr(data['normalized_layer_attention'][layer_no, :], data['fixation'])
            layer_trt_correlations_spearman[layer_no].append(trt_res.statistic)
            layer_fixation_correlations_spearman[layer_no].append(fixation_res.statistic)
            layer_trt_p_value_spearman[layer_no].append(trt_res.pvalue)
            layer_fixation_p_value_spearman[layer_no].append(fixation_res.pvalue)
            data['layer_trt_correlations_spearman'].append(trt_res.statistic)
            data['layer_fixation_correlations_spearman'].append(fixation_res.statistic)
        
            # KL-divergence
            # print(data['trt'])
            # print(data['normalized_layer_attention'][layer_no, :])
            trt_kl      = sum(rel_entr(data['trt'], data['normalized_layer_attention'][layer_no, :]))
            fixation_kl = sum(rel_entr(data['fixation'], data['normalized_layer_attention'][layer_no, :]))
            # print('KL-divergence:', trt_kl, fixation_kl)
            layer_trt_correlations_kl[layer_no].append(trt_kl)
            layer_fixation_correlations_kl[layer_no].append(fixation_kl)
            data['layer_trt_correlations_kl'].append(trt_kl)
            data['layer_fixation_correlations_kl'].append(fixation_kl)
        
        # print(layer_trt_p_value_spearman)
        # break
        # print(layer_fixation_correlations_spearman)

    layer_trt_correlations_spearman = np.array(layer_trt_correlations_spearman)
    layer_fixation_correlations_spearman = np.array(layer_fixation_correlations_spearman)
    layer_trt_p_value_spearman = np.array(layer_trt_p_value_spearman)
    layer_fixation_p_value_spearman = np.array(layer_fixation_p_value_spearman)
    layer_trt_correlations_kl = np.array(layer_trt_correlations_kl)
    layer_fixation_correlations_kl = np.array(layer_fixation_correlations_kl)
    # print(layer_trt_correlations_spearman.shape)

    ave_layer_correlations_spearman = np.mean(layer_trt_correlations_spearman + layer_fixation_correlations_spearman, axis=1)
    ave_layer_correlations_kl = np.mean(layer_trt_correlations_kl + layer_fixation_correlations_kl, axis=1)
    print('The best layer index in terms of Spearman\'s rank correlation coefficient is ', np.argmax(ave_layer_correlations_spearman))
    print('The best layer index in terms of KL-divergence is ', np.argmax(ave_layer_correlations_kl))
    print('The p-values for TRT of layers:', np.mean(layer_trt_p_value_spearman, axis=1))
    print('The p-values for fixations of layers:', np.mean(layer_fixation_p_value_spearman, axis=1))
    

    layer_correlation = {'layer_trt_correlations_spearman': np.mean(layer_trt_correlations_spearman, axis=1).tolist(), 
                         'layer_fixation_correlations_spearman': np.mean(layer_fixation_correlations_spearman, axis=1).tolist(), 
                         'layer_trt_correlations_kl': np.mean(layer_trt_correlations_kl, axis=1).tolist(),
                         'layer_fixation_correlations_kl': np.mean(layer_fixation_correlations_kl, axis=1).tolist()}
    return data_list, np.argmax(ave_layer_correlations_spearman), layer_correlation

def entry_level_correlation(data_list, layer_index):
    entry_correlation_trt_spearman = []
    entry_correlation_fixation_spearman = []
    entry_correlation_trt_kl = []
    entry_correlation_fixation_kl = []
    for data in data_list:
        entry_correlation_trt_spearman.append(data['layer_trt_correlations_spearman'][layer_index])
        entry_correlation_fixation_spearman.append(data['layer_fixation_correlations_spearman'][layer_index])
        entry_correlation_trt_kl.append(data['layer_trt_correlations_kl'][layer_index])
        entry_correlation_fixation_kl.append(data['layer_fixation_correlations_kl'][layer_index])
    
    entry_correlation = {
        "entry_correlation_trt_spearman":entry_correlation_trt_spearman,
        "entry_correlation_fixation_spearman":entry_correlation_fixation_spearman,
        "entry_correlation_trt_kl":entry_correlation_trt_kl,
        "entry_correlation_fixation_kl":entry_correlation_fixation_kl
    }
    return entry_correlation

def text_level_correlation(data_list_with_correlation_scores, data_by_sentences, layer_index):
    text_correlation = {
        "entry_correlation_trt_spearman":[],
        "entry_correlation_fixation_spearman":[],
        "entry_p_value_trt_spearman":[],
        "entry_p_value_fixation_spearman":[],
        "entry_correlation_trt_kl":[],
        "entry_correlation_fixation_kl":[]
    }
    for sentence_id in data_by_sentences:
        trt_attention = data_by_sentences[sentence_id][0]['trt']
        fixation_attention = data_by_sentences[sentence_id][0]['fixation']
        for idx in range(1, len(data_by_sentences[sentence_id])):
            trt_attention += data_by_sentences[sentence_id][idx]['trt']
            fixation_attention += data_by_sentences[sentence_id][idx]['fixation']

        trt_attention /= len(data_by_sentences[sentence_id])
        fixation_attention /= len(data_by_sentences[sentence_id])
        transformer_attention = data_by_sentences[sentence_id][0]['normalized_layer_attention'][layer_index, :]

        trt_res_spearman = stats.spearmanr(transformer_attention, trt_attention)
        fixation_res_spearman = stats.spearmanr(transformer_attention, fixation_attention)
        trt_kl = sum(rel_entr(trt_attention, transformer_attention))
        fixation_kl = sum(rel_entr(fixation_attention, transformer_attention))

        text_correlation["entry_correlation_trt_spearman"].append(trt_res_spearman.statistic)
        text_correlation["entry_correlation_fixation_spearman"].append(fixation_res_spearman.statistic)
        text_correlation["entry_p_value_trt_spearman"].append(trt_res_spearman.pvalue)
        text_correlation["entry_p_value_fixation_spearman"].append(fixation_res_spearman.pvalue)
        text_correlation["entry_correlation_trt_kl"].append(trt_kl)
        text_correlation["entry_correlation_fixation_kl"].append(fixation_kl)
        
    return text_correlation

def participant_level_correlation(data_list_with_correlation_scores, data_by_participants, layer_index):
    participant_correlation = {
        "entry_correlation_trt_spearman":[],
        "entry_correlation_fixation_spearman":[],
        "entry_p_value_trt_spearman":[],
        "entry_p_value_fixation_spearman":[],
        "entry_correlation_trt_kl":[],
        "entry_correlation_fixation_kl":[]
    }
    for participant_id in data_by_participants:
        trt_correlation_spearman = []
        fixation_correlation_spearman = []
        trt_pvalue_spearman = []
        fixation_pvalue_spearman = []
        trt_correlation_kl = []
        fixation_correlation_kl = []
        
        for entry in data_by_participants[participant_id]:
            trt_attention = entry['trt']
            fixation_attention = entry['fixation']
            transformer_attention = entry['normalized_layer_attention'][layer_index, :]

            trt_res_spearman = stats.spearmanr(transformer_attention, trt_attention)
            fixation_res_spearman = stats.spearmanr(transformer_attention, fixation_attention)
            trt_kl = sum(rel_entr(trt_attention, transformer_attention))
            fixation_kl = sum(rel_entr(fixation_attention, transformer_attention))
            
            trt_correlation_spearman.append(trt_res_spearman.statistic)
            fixation_correlation_spearman.append(fixation_res_spearman.statistic)
            trt_pvalue_spearman.append(trt_res_spearman.pvalue)
            fixation_pvalue_spearman.append(fixation_res_spearman.pvalue)
            trt_correlation_kl.append(trt_kl)
            fixation_correlation_kl.append(fixation_kl)
        

        participant_correlation["entry_correlation_trt_spearman"].append(np.mean(trt_correlation_spearman))
        participant_correlation["entry_correlation_fixation_spearman"].append(np.mean(fixation_correlation_spearman))
        participant_correlation["entry_p_value_trt_spearman"].append(np.mean(trt_pvalue_spearman))
        participant_correlation["entry_p_value_fixation_spearman"].append(np.mean(fixation_pvalue_spearman))
        participant_correlation["entry_correlation_trt_kl"].append(np.mean(trt_correlation_kl))
        participant_correlation["entry_correlation_fixation_kl"].append(np.mean(fixation_correlation_kl))
    return participant_correlation


def merge_attention(token_spans, token_attention):
    # Function to merge token-wise attention into word-wise attention
    word_spans = []
    word_attention = []
    
    # Initialize the first word
    current_word_start = token_spans[0][0]
    current_word_end = token_spans[0][1]
    current_word_attention = torch.unsqueeze(token_attention[:, 0], 1)
    current_word_count = 1
    
    for i in range(1, len(token_spans)):
        start, end = token_spans[i]
        attention = torch.unsqueeze(token_attention[:, i], 1)
        # print('shape:', attention.shape)        
        # Check if this token is part of the current word (overlapping or contiguous)
        if start <= current_word_end:  # Overlapping or contiguous
            current_word_end = max(current_word_end, end)
            current_word_attention += attention
            current_word_count += 1
        else:
            # Finalize the current word and start a new one
            word_spans.append([current_word_start, current_word_end])
            word_attention.append(current_word_attention / current_word_count)
            
            current_word_start = start
            current_word_end = end
            current_word_attention = attention
            current_word_count = 1
    
    # Append the last word
    word_spans.append([current_word_start, current_word_end])
    word_attention.append(current_word_attention / current_word_count)
    
    return word_spans, word_attention

def get_attention(context, input_text, tokenizer, model):
    input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")
    context_ids = tokenizer(context, return_tensors="pt", return_offsets_mapping=True)
    
    outputs = model.generate(**input_ids, 
        max_new_tokens=10,
        return_dict_in_generate=True,
        output_scores = False,
        output_logits = False,
        return_dict   = True,
        output_hidden_states=False,
        output_attentions = True,
        )
    # print(tokenizer.decode(outputs[0][0]))
    # print(tokenizer.decode(outputs.sequences[0]), outputs.sequences.shape) # torch.Size([1, 144])
    averaged_layer_attention = []
    for layer_attention in outputs.attentions[0]: # 第一个元素存放了之前的所有注意力，后续的token只存放他们自己的注意力
        averaged_layer_attention.append(torch.mean(layer_attention[..., -1, :], dim=1)) # average along heads
    # extract attention scores on the text span
    averaged_layer_attention = torch.cat(averaged_layer_attention, 0)[:, 1:context_ids['input_ids'].shape[1]]
    print('context: ', tokenizer.decode(context_ids['input_ids'][0]))
    print(averaged_layer_attention.shape)
    context_tokens = tokenizer.tokenize(context)
    context_text_spans = context_ids['offset_mapping'][0][1:].tolist()
    remove_idices = []
    for i, token in enumerate(context_tokens):
        if 1 == len(token) and 9601 == ord(token[0]):
            remove_idices.append(i)
        elif token.startswith(chr(9601)):
            context_text_spans[i][0] += 1
        # print(token, ord(token[0]))
    # print('Before processing:')
    # print(context_tokens)
    context_indices = [i for i in range(len(context_tokens)) if i not in remove_idices]
    # print("remove_idices:", remove_idices)
    # print('After processing:')
    context_text_spans = [token for i, token in enumerate(context_text_spans) if i not in remove_idices]
    context_tokens = [token for i, token in enumerate(context_tokens) if i not in remove_idices]
    averaged_layer_attention = averaged_layer_attention[:, context_indices].contiguous()
    # print(context_tokens)
    # print(context_text_spans)
    # print('No. of tokens:', len(context_text_spans))
    # Merge into words' attention
    # for span in context_text_spans:
    word_spans, word_attention = merge_attention(context_text_spans, averaged_layer_attention)
    print('words: ', " ".join([context[span[0]: span[1]] for span in word_spans]))
    # print('word_attention', len(word_attention))
    merged_attention = torch.cat(word_attention, 1)
    # normalization
    normalized_layer_attention = merged_attention / torch.sum(merged_attention, dim=1, keepdim=True)
    print(normalized_layer_attention.shape) 
    return normalized_layer_attention.detach().cpu().float().numpy()