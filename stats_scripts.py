import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


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
                ax.set_ylabel('Layer')
                ax.set_xlabel('Head')

                ax.set_xticks(np.arange(0, 12))
                ax.set_yticks(np.arange(0, 12))
                ax.set_xticklabels(np.arange(1, 13))
                ax.set_yticklabels(np.arange(1, 13))
            elif mode == 'importance':
                cmap = plt.cm.get_cmap('plasma')
                normalized_y = 1 - (tensor[0]) / (12)
                ax.bar(tensor[1],tensor[0],color=cmap(normalized_y)) # barplot with layers with most used heads
                ax.set_ylabel('Heads in use')
                ax.set_xlabel('Layer')
                ax.set_xticks(np.arange(0, 12))
                ax.set_yticks(np.arange(0, 13))
                ax.set_xticklabels(np.arange(1, 13))
                #ax.set_yticklabels(np.arange(1, 13))
            ax.set_title(f'Task: {key}, Language: {inner_key}')  
            col += 1
        row += 1
        col = 0
    plt.tight_layout() 
    plt.show()
  


def determine_overlap(ratio_dict,task1,lang1,task2,lang2):
    '''
    Returns:
        overlap_percentage: percentage of common values for the two ratio matrices
        Also plots the overlap.
    '''
    tensor1 = ratio_dict[task1][lang1]
    tensor2 = ratio_dict[task2][lang2]
    eq = torch.eq(tensor1 , tensor2)
    overlap_count = torch.sum(eq).item()
    total_elements = tensor1.numel()
    overlap_percentage = (overlap_count / total_elements) * 100
    print(f'The overlap percentage for {task1,lang1} and  {task2,lang2} is {overlap_percentage:.2f}%' )  

    _, ax = plt.subplots()
    ax.imshow(eq, cmap='Blues')  # heatmap for masks
    ax.set_ylabel('Layer')
    ax.set_xlabel('Head')
    ax.set_title(f'Overlap for {task1,lang1} and  {task2,lang2}')  
    ax.set_xticks(np.arange(0, 12))
    ax.set_yticks(np.arange(0, 12))
    ax.set_xticklabels(np.arange(1, 13))
    ax.set_yticklabels(np.arange(1, 13))
    plt.show()
    return overlap_percentage


if __name__ == '__main__':
    masks_dict = load_masks()
    ratio_dict, importance_dict, layer_importance_dict = determine_importance(masks_dict)        
    visualize_dictionaries(layer_importance_dict,'importance')
    overlap = determine_overlap(ratio_dict,'xnli','en','xnli','de')



