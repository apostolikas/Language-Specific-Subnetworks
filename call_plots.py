from make_plots import plot_square_matrix
import cv2 #make will be removed for the final version
from stats_scripts import load_masks, determine_importance

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

def create_big_plot(images):
    '''
    images : list of list 
    '''
    im_tile = cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in images])
    cv2.imwrite('survivors.jpg', im_tile)

if __name__ == '__main__':
    masks_dict = load_masks()
    images = show_survival_prob_plots(masks_dict)
    create_big_plot(images)
