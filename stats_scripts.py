import torch
import os


def jaccard_similarity_triplet(x1, x2, x3, x4, x5, n):
    if n == 3:
        intersection = x1.intersection(x2, x3)
        union = x1.union(x2, x3)
    elif n==5:
        intersection = x1.intersection(x2, x3, x4, x5)
        union = x1.union(x2, x3, x4, x5)
    else:
        raise Exception("Choose either 3 or 5 (tasks & languages)")
    similarity = len(intersection) / len(union)
    return similarity


LANGUAGES = ['en','de','fr','es','zh']
TASKS = ['marc','paws-x','xnli']
NUM_SEEDS = 5
SEEDS = [i for i in range(NUM_SEEDS)]

def load_masks():
    '''
    load all masks.pkl files

    Returns:
        dict: dict of the form {language: { task: {seed: 2d_mask}}}
    '''
    mask_dict = {}
    for lang in LANGUAGES:
        mask_dict[lang] = {}
        for task in TASKS:
            mask_dict[lang][task] = {}
            for seed in range(NUM_SEEDS):
                path = os.path.join('./results/pruned_masks', task, f'{lang}_{seed}.pkl')
                mask = torch.load(path, map_location='cpu')
                mask_dict[lang][task][seed] = mask
    return mask_dict


def determine_importance(masks_dict:dict, lang:str, task:str) -> torch.Tensor:
  
    count_matrix = torch.zeros((12,12))
    torch.set_printoptions(precision=1)
    for _,mask_values in masks_dict[lang][task].items():
        count_matrix += mask_values
    ratio_matrix = count_matrix/len(masks_dict[lang][task].keys())

    _, sorted_layers = torch.sum(ratio_matrix==0, axis=1).sort() # importance in descending order based on the occurences of zeros

    most_important_heads_tensor = torch.argwhere(ratio_matrix==torch.max(ratio_matrix))
    layer_loc_of_important_head = most_important_heads_tensor[:,0]
    place_of_important_head = most_important_heads_tensor[:,1]

    for layer,place in zip(layer_loc_of_important_head, place_of_important_head):
        print('One of the most important heads is the #',place.item()+1, 'in layer',layer.item()+1) # All the printed heads have the same ratio (i.e. 1 - always on)
    print('Importance of layers in descending order :' ,[x.item()+1 for x in sorted_layers])
    print('The matrix showing the ratio of how much is used each head\n',ratio_matrix)

    return ratio_matrix


def find_layer_patterns(masks_dict:dict) -> dict:
    
    language_pattern_dict = {}
    pattern_dict = {}

    for task in TASKS:
        for lang in LANGUAGES:

            count_matrix = torch.zeros((12,12))

            for _,mask_values in masks_dict[lang][task].items():
                count_matrix += mask_values
            _, sorted_layers = torch.sum(count_matrix==0, axis=1).sort() # axis = 1 if layer = row

            language_pattern_dict[lang] = sorted_layers
        pattern_dict[task] = language_pattern_dict

    #! Pointless sim always = 1. Must do low level comparison (mask-wise)
    # Layer-wise comparison (not mask-wise per se) - more high level comparison
    # for lang in LANGUAGES:
    #     print('For', lang, ':', jaccard_similarity_triplet(set(pattern_dict['marc'][lang].tolist()),set(pattern_dict['paws-x'][lang].tolist()),set(pattern_dict['xnli'][lang].tolist()),_,_,3))

    # for task in TASKS:
    #     print('For', task, ':', jaccard_similarity_triplet(set(pattern_dict[task]['en'].tolist()),
    #                                                        set(pattern_dict[task]['de'].tolist()),
    #                                                        set(pattern_dict[task]['fr'].tolist()),
    #                                                        set(pattern_dict[task]['es'].tolist()),
    #                                                        set(pattern_dict[task]['zh'].tolist()),
    #                                                        5))

    for task in pattern_dict.keys():
        print('\nFor',task)
        for lang in language_pattern_dict.keys():
            print('For', lang, 'the layer importance is' ,pattern_dict[task][lang])

    return pattern_dict


masks_dict = load_masks()
ratio_matrix = determine_importance(masks_dict,'en','marc')
pattern_dict = find_layer_patterns(masks_dict)

