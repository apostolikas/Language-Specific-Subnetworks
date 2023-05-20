from make_plots import plot_square_matrix, create_big_plot
import cv2 #make will be removed for the final version
from stats_scripts import load_masks, determine_importance
import numpy as np
import os
LANGUAGES = ['en','de','fr','es','zh']
TASKS = ['marc','paws-x','xnli']

def show_survival_prob_plots(masks_dict):
    ratio_dict, _, _ = determine_importance(masks_dict)
    os.makedirs('images/survivor_probability/', exist_ok=True)
    images = []
    for task in TASKS:
        cur_task_images = []
        for lang in LANGUAGES:
            matrix = ratio_dict[task][lang].numpy()
            x_labels = [i for i in range(matrix.shape[0])]
            y_labels = [i for i in range(matrix.shape[1])]
            save_path =  os.path.join('./images', 'survivor_probability', 
                                      f'{task}_{lang}')
            #f'survivor_prob_{task}_{lang}'
            plot_square_matrix(matrix, x_labels, y_labels, save_path = save_path, x_title='Head', y_title='Layer',
                            title = f'survivor_probability_{task}_{lang}')
            img = cv2.imread(save_path+'.png')
            cur_task_images.append(img)
        images.append(cur_task_images)
    return images

def show_timesteps_head_scores(all_steps_head_scores, last_step_head_scores_info):
    '''
    same language, same task show in a big figure all head scores for the first 4 pruning steps and all seeds
    '''
    cur_big_plt_images = []
    path = os.path.join('./images', 'timesteps')
    os.makedirs(path, exist_ok=True)
    for i, tmp_all_steps_scores in enumerate(all_steps_head_scores):
        task, lang, seed = last_step_head_scores_info[i]
        cur_task_images = []
        for t, cur_step_scores in enumerate(tmp_all_steps_scores): # for all pruning timesteps
            cur_step_scores = cur_step_scores.detach().cpu().numpy()
            x_labels = [i for i in range(cur_step_scores.shape[0])]
            y_labels = [i for i in range(cur_step_scores.shape[1])]
            title = f'{task}_{lang}_seed{seed}_time{t}'
            save_path = os.path.join('./images', 'timesteps',title)
            x_title='Head'
            y_title='Layer'
           
            plot_square_matrix(cur_step_scores, x_labels, y_labels, save_path, x_title, y_title, title,
                                colour_map='binary', change_colors=True)
            if t <= 3: #see only the first 4 pruning timesteps
                img = cv2.imread(save_path+'.png')
                cur_task_images.append(img)
        
        for i in range(4-len(cur_task_images)):
            dummy_img = np.ones_like(cur_task_images[-1])
            cur_task_images.append(dummy_img)

        cur_big_plt_images.append(cur_task_images)
        if seed==4: #finished adding all seeds so now we can create the big plot
            create_big_plot(cur_big_plt_images,f'big_timesteps3_{task}_{lang}.jpg')
            cur_big_plt_images = []

if __name__ == '__main__':
    masks_dict = load_masks()
    images = show_survival_prob_plots(masks_dict)
    create_big_plot(images,'survivors.jpg')
