'''
Download gergopool masks from this link https://www.transferxl.com/download/08d4Q1jw024QT
after unzipping you should have a folder called results
'''
import numpy as np
import torch
import itertools
import statistics
import json
import matplotlib.pyplot as plt

from make_plots import plot_lower_triangular_matrix
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
                path = f'results\pruned_masks\{task}\{lang}_{seed}.pkl'
                mask = torch.load(path)
                mask_dict[lang][task][seed] = mask
    return mask_dict

def compute_jaccard_similarity(mask_1, mask_2):
    '''
    Parameters:
        masks_dict: dict of the form {language: { task: {seed: 2d_mask}}}
        lang_1: string language 1
        lang_2: string language 2
        task_1: string task 1
        task_2: string task 2
        seed: int 
    Returns:
        jaccard_sim: int
    '''
    intersect = len([head_1 for head_1,head_2 in zip(mask_1,mask_2) if head_1==head_2==1])
    union = len([head_1 for head_1,head_2 in zip(mask_1,mask_2) if head_1==1 or head_2==1])
    jaccard_sim = intersect/union

    return jaccard_sim

def jaccard_similarity_per_task_pair(masks_dict):
    '''
    same language, take tasks pair and calculate mean,std across seeds

    Parameters:
        masks_dict: dict of the form {language: { task: {seed: 2d_mask}}}
    Returns:
        stats_jaccard_task_dict: dict of the form {language: { (task1, task2) : 'mean_jaccard+-std_jaccard'}}
    '''

    stats_jaccard_task_dict = {}# mean+-std across seeds
    max_value = -1
    print('TASK PAIRS')
    for lang in LANGUAGES:

        stats_jaccard_task_dict[lang] = {}

        for task_1, task_2 in itertools.combinations(TASKS, r=2):
            tmp_list_jaccard_seeds = []

            for seed in range(NUM_SEEDS):
                #! pass same language, same seed but different tasks
                mask_1 = masks_dict[lang][task_1][seed].flatten().tolist()
                mask_2 = masks_dict[lang][task_2][seed].flatten().tolist()

                tmp_jaccard_sim = compute_jaccard_similarity(mask_1, mask_2)
                tmp_list_jaccard_seeds.append(tmp_jaccard_sim)

            mean = round(statistics.mean(tmp_list_jaccard_seeds),3)
            std = round(statistics.stdev(tmp_list_jaccard_seeds),3)
            stats_jaccard_task_dict[lang][(task_1,task_2)] = f'{mean}\u00B1{std}'# mean +- std
            if max_value < mean:
                mean = max_value
                t1,t2 = task_1, task_2
    print('MEAN ', mean)
    print(t1, ' ',t2)

    return stats_jaccard_task_dict

def jaccard_similarity_per_lang_pair(masks_dict):
    '''
        same task take language pairs 

        Parameters:
            masks_dict: dict of the form {language: { task: {seed: 2d_mask}}}
        Returns:
            stats_jaccard_lang_dict: dict of the form {task: { (language1, language2): 'mean_jaccard+-std_jaccard'}}
    '''
    stats_jaccard_lang_dict = {}# mean+-std across seeds
    print('LANGUAGE PAIRS')
    max_value = -1
    for task in TASKS:

        stats_jaccard_lang_dict[task] = {}

        for lang_1, lang_2 in itertools.combinations(LANGUAGES, r=2):
            tmp_list_jaccard_seeds = []
            for seed in range(NUM_SEEDS):
                #! pass same task, same seed but different languages
                mask_1 = masks_dict[lang_1][task][seed].flatten().tolist()
                mask_2 = masks_dict[lang_2][task][seed].flatten().tolist()

                tmp_jaccard_sim = compute_jaccard_similarity(mask_1, mask_2)
                tmp_list_jaccard_seeds.append(tmp_jaccard_sim)

            mean = round(statistics.mean(tmp_list_jaccard_seeds),3)
            std = round(statistics.stdev(tmp_list_jaccard_seeds),3)
            stats_jaccard_lang_dict[task][(lang_1,lang_2)] = f'{mean}\u00B1{std}'# mean +- std
            if max_value < mean:
                mean = max_value
                t1,t2 = lang_1, lang_2
    print('MEAN ', mean)
    print(t1, ' ',t2)

    return stats_jaccard_lang_dict

def jaccard_similarity_per_seed_pair(masks_dict):
    '''
        same task same language, different seed 
        Parameters:
            masks_dict: dict of the form {language: { task: {seed: 2d_mask}}}
        Returns:
            stats_jaccard_seed_dict: dict of the form {task: {language: {(seed_i, seed_j) : 'jaccard'}}}
    '''
    jaccard_seed_dict = {} 
    stats_jaccard_seed_dict = {}
    for task in TASKS:
        jaccard_seed_dict[task] = {}
        stats_jaccard_seed_dict[task] = {}

        for lang in LANGUAGES:
            jaccard_seed_dict[task][lang] = {}
            tmp_list_jaccard_seeds = []
            for seed_i, seed_j in itertools.combinations(SEEDS, r=2):
                #! pass same task, same language but different seeds
                mask_1 = masks_dict[lang][task][seed_i].flatten().tolist()
                mask_2 = masks_dict[lang][task][seed_j].flatten().tolist()
                tmp_jaccard_sim = compute_jaccard_similarity(mask_1, mask_2)

                jaccard_seed_dict[task][lang][(seed_i,seed_j)] = tmp_jaccard_sim
                tmp_list_jaccard_seeds.append(tmp_jaccard_sim)

            mean = round(statistics.mean(tmp_list_jaccard_seeds),3)
            std = round(statistics.stdev(tmp_list_jaccard_seeds),3)
            stats_jaccard_seed_dict[task][lang] = f'{mean}\u00B1{std}'# mean +- std

    return stats_jaccard_seed_dict


def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

def surviving_per_task(masks_dict):
    '''
    create super-survivors for every language 
    overlap_lang(i,i): number of surviving heads common for all seeds for task i
    overlap_langll(i,j): intersect(number of surviving heads common for all seeds for task i, number of 
        surviving heads common for all seeds for task j)
    '''
    for lang in LANGUAGES:
        heads_ids = []
        for task in TASKS:
            cur_task_heads_ids = []
            combined_iterable = zip(*[masks_dict[lang][task][seed].flatten().tolist() for seed in SEEDS])
            for i, masks_tuple in enumerate(combined_iterable):
                if all(mask_seed == 1.0 for mask_seed in masks_tuple):
                    cur_task_heads_ids.append(i)
            heads_ids.append(cur_task_heads_ids)

        overlap_lang = np.zeros((len(TASKS),len(TASKS)))
        for task_id in range(len(TASKS)):
            overlap_lang[task_id, task_id] = len(heads_ids[task_id])
        for task_id1, task_id2 in itertools.combinations([k for k in range(len(TASKS))], r=2):
            overlap_lang[task_id1, task_id2] =  len(intersection(heads_ids[task_id1], heads_ids[task_id2]))
        plot_lower_triangular_matrix(overlap_lang, TASKS, TASKS, f'super_survivors_{lang}', 'TASK', 'TASK')


masks_dict = load_masks()
surviving_per_task(masks_dict)
jaccard_task_dict = jaccard_similarity_per_task_pair(masks_dict)
jaccard_lang_dict = jaccard_similarity_per_lang_pair(masks_dict)
jaccard_seeds_dict = jaccard_similarity_per_seed_pair(masks_dict)

A=1
# with open('jaccard_task_dict.json', 'w') as file:
#      json.dumps(jaccard_task_dict)