import matplotlib.pyplot as plt
import numpy as np
import cv2 #maybe this will be removed for the final version

# settings
LANGUAGES = ['en','de','fr','es','zh']
TASKS = ['marc','paws-x','xnli']
NUM_SEEDS = 5
SEEDS = [i for i in range(NUM_SEEDS)]

def plot_lower_triangular_matrix(matrix, x_labels, y_labels, save_path, x_title, y_title, colour_map=plt.cm.OrRd):
    initial_matrix = matrix

    mask =  np.tri(matrix.shape[0], k=-1).T
    matrix = np.ma.array(matrix, mask=mask)

    fig, ax = plt.subplots(figsize=(9,8))
    ax.set_xlabel(x_title, labelpad=20)
    ax.set_ylabel(y_title)
    ax.xaxis.set_ticks_position('bottom')
    ax.xaxis.set_label_position('bottom')
    ax.set_xticks(range(len(x_labels)))
    ax.set_yticks(range(len(y_labels)))
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    mask =  np.tri(matrix.shape[0], k=0).T # Leave Diagonal Uncoloured
    mean_matrix_wo_diagonal = np.ma.array(matrix, mask=mask) 
    
    ax.imshow(mean_matrix_wo_diagonal, cmap=colour_map)
    plt.gca().xaxis.tick_bottom()
    
    
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if i <= j:
                mean = initial_matrix[i, j]
                text = f"{mean:.2f}"
                # color = 'black' if mean < mean_mean or i == j else 'white'
                ax.text(i, j, text, va='center', ha='center', color='black')
    if save_path:
        fig.savefig(save_path)

def plot_square_matrix(matrix, x_labels, y_labels, save_path, x_title, y_title, title, colour_map=plt.cm.OrRd,
                       change_colors=False):
    '''
    matrix should be a numpy array
    '''
    fig, ax = plt.subplots(figsize=(9,8))
    ax.set_xlabel(x_title, labelpad=20)
    ax.set_ylabel(y_title)
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels)
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels)
    ax.imshow(matrix, cmap=colour_map)
    ax.set_title(title)
    # plt.tight_layout()
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            cell_value = matrix[i,j]
            if not change_colors:
                ax.text(j, i, f"{cell_value:.2f}\n", va='center', ha='center')
            else:
                if cell_value > 0:
                    color = 'black'
                else:
                    color = 'white'
                ax.text(j, i, f"{cell_value:.2f}\n", va='center', ha='center', color=color)
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    plt.close()
    # maybe we can remove also the titles I am not that it is a good idea
    # plt.title(title)
    # plt.show()

def create_big_plot(images,plt_name):
    '''
    images : list of list 
    '''
    im_tile = cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in images])
    cv2.imwrite(plt_name, im_tile)

def plot_tSNE(tsne_output, head_scores_info):
    LANGUAGES = ['en','de','fr','es','zh']
    TASKS = ['marc','paws-x','xnli']
    colours = ['red','green','blue'] # mark tasks with colours
    markers = ['^','o','*','X','s'] # mark languages with plus, circle, *, X, square

    dict_task_colors = {task: colours[i] for i, task in enumerate(TASKS)}
    dict_language_markers = {lang: markers[i] for i, lang in enumerate(LANGUAGES)}

    for point, (task, language,_) in zip(tsne_output, head_scores_info):
        plt.scatter(point[0], point[1], color=dict_task_colors[task], marker=dict_language_markers[language])

    # # Add legend for colors
    # color_legends = [plt.Line2D([], [], linestyle='None',marker='o', color=dict_task_colors[task], markersize=8)
    #                 for task in TASKS]
    # plt.legend(color_legends, TASKS, loc='lower left')
    # Add legend for colors
    # color_legends = [plt.Line2D([], [], linestyle='None', marker='o', color=dict_task_colors[task], markersize=8)
    #                 for task in TASKS]

    # marker_legends = [plt.Line2D([], [], linestyle='None', marker=dict_language_markers[lang], color='black', markersize=8)
    #               for lang in LANGUAGES]

    # # Plot the legends
    # plt.legend(color_legends, TASKS, loc='lower left', bbox_to_anchor=(0, 1))
    # plt.legend(marker_legends, LANGUAGES, loc='upper right', bbox_to_anchor=(1, 1))

    # # Add the legends to the plot without overlapping
    # plt.gca().add_artist(plt.legend(color_legends, TASKS, loc='lower left', bbox_to_anchor=(0, 1)))
    # plt.gca().add_artist(plt.legend(marker_legends, LANGUAGES, loc='upper right', bbox_to_anchor=(1, 1)))
    legends = []
    legend_str = []
    for task in TASKS:
        for lang in LANGUAGES:
            legends.append(plt.Line2D([], [], linestyle='None', marker=dict_language_markers[lang],
                                    color=dict_task_colors[task], markersize=8))
            legend_str.append(f'{task}_{lang}')

    # plt.legend(legends, legend_labels, loc='best', ncol=3)
    plt.legend(legends, legend_str, loc='center', bbox_to_anchor=(1.2, 0.5))

    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.title("t-SNE Visualization")
    plt.tight_layout()
    plt.show()