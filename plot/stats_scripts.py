import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import itertools

import sys
if "./" not in sys.path:
    sys.path.append("./")

from data import ALLOWED_LANGUAGES, ALLOWED_DATASETS

LANGUAGES = ALLOWED_LANGUAGES
TASKS = ALLOWED_DATASETS
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


def determine_importance(masks_dict: dict) -> torch.Tensor:
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
            count_matrix = torch.zeros((12, 12))
            torch.set_printoptions(precision=1)
            for _, mask_values in masks_dict[lang][task].items():
                count_matrix += mask_values
            ratio_matrix = count_matrix / len(masks_dict[lang][task].keys())
            most_important_heads_tensor = torch.argwhere(ratio_matrix == torch.max(ratio_matrix))
            ratio_dict[task][lang] = ratio_matrix
            importance_dict[task][lang] = most_important_heads_tensor
            layer_importance_dict[task][lang] = torch.sum(ratio_matrix != 0, axis=1).sort()
    return ratio_dict, importance_dict, layer_importance_dict


def visualize_dictionaries(input_dict: dict, mode: str) -> None:
    '''
    Plots the content of a nested dicts (1st lang - 2nd task)
    '''
    _, axs = plt.subplots(4, 5, figsize=(12, 12))
    row = 0
    col = 0
    for key, inner_dict in input_dict.items():
        if row >= 4:
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
                ax.bar(tensor[1], tensor[0],
                       color=cmap(normalized_y))  # barplot with layers with most used heads
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
    plt.savefig('results/plots/appendix_' + mode + '_fig.pdf')
    plt.show()


def determine_overlap(mask_dict: dict, task: str, lang: str, mode: str) -> None:
    '''
    Prints:
        percentage of overlap langwise or taskwise
    '''

    if mode == 'l':
        seeds_dict = {}
        for seed in SEEDS:
            for i, (lang1, lang2) in enumerate(itertools.combinations(LANGUAGES, r=2)):
                tensor1 = mask_dict[lang1][task][seed]
                tensor2 = mask_dict[lang2][task][seed]
                eq = torch.eq(tensor1, tensor2)
                overlap_count = torch.sum(eq).item()
                total_elements = tensor1.numel()
                overlap_percentage = (overlap_count / total_elements) * 100
                if (lang1, lang2) in seeds_dict:
                    seeds_dict[(lang1, lang2)] += overlap_percentage
                else:
                    seeds_dict[(lang1, lang2)] = overlap_percentage
        for key in seeds_dict:
            seeds_dict[key] /= NUM_SEEDS
            print(
                f'The overlap percentage for {task, key[0]} and {task, key[1]} is {seeds_dict[key]:.2f}%'
            )

    elif mode == 't':
        seeds_dict = {}
        for seed in SEEDS:
            for i, (task1, task2) in enumerate(itertools.combinations(TASKS, r=2)):
                tensor1 = mask_dict[lang][task1][seed]
                tensor2 = mask_dict[lang][task2][seed]
                eq = torch.eq(tensor1, tensor2)
                overlap_count = torch.sum(eq).item()
                total_elements = tensor1.numel()
                overlap_percentage = (overlap_count / total_elements) * 100

                if (task1, task2) in seeds_dict:
                    seeds_dict[(task1, task2)] += overlap_percentage
                else:
                    seeds_dict[(task1, task2)] = overlap_percentage
        for key in seeds_dict:
            seeds_dict[key] /= NUM_SEEDS
            print(
                f'The overlap percentage for {lang, key[0]} and {lang, key[1]} is {seeds_dict[key]:.2f}%'
            )


if __name__ == '__main__':
    masks_dict = load_masks()
    ratio_dict, importance_dict, layer_importance_dict = determine_importance(masks_dict)
    visualize_dictionaries(ratio_dict, 'mask')
    visualize_dictionaries(layer_importance_dict, 'importance')
    print('Overlap of subnetworks for the same task across different languages')
    for task in TASKS:
        determine_overlap(masks_dict, task, None, 'l')
    print('\n\n')
    print('Overlap of subnetworks for the same language across different tasks')
    for lang in LANGUAGES:
        determine_overlap(masks_dict, None, lang, 't')
