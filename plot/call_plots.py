import numpy as np
import os

import sys
if "./" not in sys.path:
    sys.path.append("./")

from data import ALLOWED_LANGUAGES, ALLOWED_DATASETS
from utils.make_plots import plot_square_matrix
from plot.stats_scripts import load_masks, determine_importance

LANGUAGES = ALLOWED_LANGUAGES
TASKS = ALLOWED_DATASETS
SAVE_FOLDER = "results/plots"


def save_survival_prob_plots(masks_dict):
    ratio_dict, _, _ = determine_importance(masks_dict)
    os.makedirs(f'{SAVE_FOLDER}/survivor_probability/', exist_ok=True)
    for task in TASKS:
        for lang in LANGUAGES:
            matrix = ratio_dict[task][lang].numpy()
            x_labels = [i for i in range(matrix.shape[0])]
            y_labels = [i for i in range(matrix.shape[1])]
            save_path = os.path.join(SAVE_FOLDER, 'survivor_probability', f'{task}_{lang}')
            #f'survivor_prob_{task}_{lang}'
            plot_square_matrix(matrix,
                               x_labels,
                               y_labels,
                               save_path=save_path,
                               x_title='Head',
                               y_title='Layer',
                               title=f'survivor_probability_{task}_{lang}')

def show_timesteps_head_scores(all_steps_head_scores, last_step_head_scores_info):
    '''
    same language, same task show in a big figure all head scores for the first 4 pruning steps and all seeds
    '''
    path = os.path.join(SAVE_FOLDER, 'timesteps')
    os.makedirs(path, exist_ok=True)
    for i, tmp_all_steps_scores in enumerate(all_steps_head_scores):
        task, lang, seed = last_step_head_scores_info[i]
        for t, cur_step_scores in enumerate(tmp_all_steps_scores):  # for all pruning timesteps
            cur_step_scores = cur_step_scores.detach().cpu().numpy()
            x_labels = [i for i in range(cur_step_scores.shape[0])]
            y_labels = [i for i in range(cur_step_scores.shape[1])]
            title = f'{task}_{lang}_seed{seed}_time{t}'
            save_path = os.path.join(SAVE_FOLDER, 'timesteps', title)
            x_title = 'Head'
            y_title = 'Layer'

            plot_square_matrix(cur_step_scores,
                               x_labels,
                               y_labels,
                               save_path,
                               x_title,
                               y_title,
                               title,
                               colour_map='binary',
                               change_colors=True)

if __name__ == '__main__':
    masks_dict = load_masks()
    save_survival_prob_plots(masks_dict)
