'''
Download gergopool masks from this link https://www.transferxl.com/download/08d4Q1jw024QT
after unzipping you should have a folder called results
'''

import torch
import itertools
import statistics

LANGUAGES = ['en','de','fr','es','zh']
TASKS = ['marc','paws-x','xnli']
NUM_SEEDS = 5

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

def compute_jaccard_similarity(masks_dict, lang_1, lang_2, task_1, task_2, seed):
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
    mask_1 = masks_dict[lang_1][task_1][seed].flatten().tolist()
    mask_2 = masks_dict[lang_2][task_2][seed].flatten().tolist()
    intersect = len([head_1 for head_1,head_2 in zip(mask_1,mask_2) if head_1==head_2==1])
    union = len([head_1 for head_1,head_2 in zip(mask_1,mask_2) if head_1==1 or head_2==1])
    jaccard_sim = intersect/union
    return jaccard_sim

def jaccard_similarity_per_task_pair(masks_dict):
    '''
    same language, take tasks pair

    Parameters:
        masks_dict: dict of the form {language: { task: {seed: 2d_mask}}}
    Returns:
        stats_jaccard_task_dict: dict of the form {language: { (task1, task2) : 'mean_jaccard+-std_jaccard'}}
    '''
    jaccard_task_dict = {} #for every seed different dict
    stats_jaccard_task_dict = {}# mean+-std across seeds
    for lang in LANGUAGES:
        jaccard_task_dict[lang] = {}
        stats_jaccard_task_dict[lang] = {}

        for task_1, task_2 in itertools.combinations(TASKS, r=2):
            tmp_list_jaccard_seeds = []
            jaccard_task_dict[lang][(task_1,task_2)] = {}
            for seed in range(NUM_SEEDS):
                #! pass same language but different tasks
                tmp_jaccard_sim = compute_jaccard_similarity(masks_dict, lang, lang, task_1, task_2, seed)

                jaccard_task_dict[lang][(task_1,task_2)][seed] = tmp_jaccard_sim
                tmp_list_jaccard_seeds.append(tmp_jaccard_sim)

            mean = round(statistics.mean(tmp_list_jaccard_seeds),3)
            std = round(statistics.stdev(tmp_list_jaccard_seeds),3)
            stats_jaccard_task_dict[lang][(task_1,task_2)] = f'{mean}\u00B1{std}'# mean +- std

    return stats_jaccard_task_dict

def jaccard_similarity_per_lang_pair(masks_dict):
    '''
        same task take a language pairs 

        Parameters:
            masks_dict: dict of the form {language: { task: {seed: 2d_mask}}}
        Returns:
            stats_jaccard_lang_dict: dict of the form {task: { (language1, language2): 'mean_jaccard+-std_jaccard'}}
    '''
    jaccard_lang_dict = {} #for every seed different dict
    stats_jaccard_lang_dict = {}# mean+-std across seeds
    for task in TASKS:
        jaccard_lang_dict[task] = {}
        stats_jaccard_lang_dict[task] = {}

        for lang_1, lang_2 in itertools.combinations(LANGUAGES, r=2):
            tmp_list_jaccard_seeds = []
            jaccard_lang_dict[task][(lang_1,lang_2)] = {}
            for seed in range(NUM_SEEDS):
                #! pass same task but different languages
                tmp_jaccard_sim = compute_jaccard_similarity(masks_dict, lang_1, lang_2, task, task, seed)

                jaccard_lang_dict[task][(lang_1,lang_2)][seed] = tmp_jaccard_sim
                tmp_list_jaccard_seeds.append(tmp_jaccard_sim)

            mean = round(statistics.mean(tmp_list_jaccard_seeds),3)
            std = round(statistics.stdev(tmp_list_jaccard_seeds),3)
            stats_jaccard_lang_dict[task][(lang_1,lang_2)] = f'{mean}\u00B1{std}'# mean +- std

    return stats_jaccard_lang_dict


masks_dict = load_masks()
jaccard_task_dict = jaccard_similarity_per_task_pair(masks_dict)
jaccard_lang_dict = jaccard_similarity_per_lang_pair(masks_dict)
a=1