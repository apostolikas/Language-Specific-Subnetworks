import torch
import os
import matplotlib.pyplot as plt

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


def determine_importance(masks_dict:dict) -> torch.Tensor:
    '''
    Returns:
        ratio_dict: how many times a head is turned off or not for every language & task
        importance_dict: location of the most important heads for every language & task in the form (layer,head_index)
        layer_importance_dict: pair of the layer with the most used heads and the least used heads
    '''

    ratio_dict = {}
    importance_dict = {}
    layer_importance_dict = {}
    for task in TASKS:
        ratio_dict[task] = {}
        importance_dict[task] = {}
        layer_importance_dict[task] = {}
        for lang in LANGUAGES:
            count_matrix = torch.zeros((12,12))
            torch.set_printoptions(precision=1)
            for _,mask_values in masks_dict[lang][task].items():
                count_matrix += mask_values
            ratio_matrix = count_matrix/len(masks_dict[lang][task].keys())
            most_important_heads_tensor = torch.argwhere(ratio_matrix==torch.max(ratio_matrix))
            ratio_dict[task][lang] = ratio_matrix
            importance_dict[task][lang] = most_important_heads_tensor
            layer_importance_dict[task][lang] = torch.sum(ratio_matrix!=0, axis=1).sort(descending=True)
    return ratio_dict, importance_dict, layer_importance_dict


def visualize_dictionaries(input_dict:dict, mode:str):
    '''
    Plots the content of a nested dicts (1st lang - 2nd task)
    '''
    _, axs = plt.subplots(3, 5, figsize=(8, 8))  
    row = 0
    col = 0
    for key, inner_dict in input_dict.items():
        if row >= 3:
            break
        for inner_key, tensor in inner_dict.items():
            if col >= 5:
                break
            ax = axs[row, col]  
            if mode == 'mask':
                ax.imshow(tensor, cmap='Blues')  # heatmap for masks
            elif mode == 'importance':
                ax.bar(tensor[1]+1,tensor[0]) # barplot with layers with most used heads
            ax.set_title(f'Task: {key}, Language: {inner_key}')  
            col += 1
        row += 1
        col = 0
    plt.tight_layout() 
    plt.show()  



masks_dict = load_masks()

ratio_dict, importance_dict, layer_importance_dict = determine_importance(masks_dict)        
visualize_dictionaries(layer_importance_dict,'importance')
