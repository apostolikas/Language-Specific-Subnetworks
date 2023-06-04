import matplotlib.pyplot as plt
import numpy as np
from data import ALLOWED_DATASETS, ALLOWED_LANGUAGES
import os
# settings
LANGUAGES = ALLOWED_LANGUAGES
NUM_SEEDS = 5
SEEDS = [i for i in range(NUM_SEEDS)]


def plot_lower_triangular_matrix(matrix,
                                 x_labels,
                                 y_labels,
                                 save_path,
                                 x_title,
                                 y_title,
                                 colour_map=plt.cm.OrRd):
    initial_matrix = matrix

    mask = np.tri(matrix.shape[0], k=-1).T
    matrix = np.ma.array(matrix, mask=mask)

    fig, ax = plt.subplots(figsize=(9, 8))
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

    mask = np.tri(matrix.shape[0], k=0).T  # Leave Diagonal Uncoloured
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


def plot_square_matrix(matrix,
                       x_labels,
                       y_labels,
                       save_path,
                       x_title,
                       y_title,
                       title,
                       colour_map=plt.cm.OrRd,
                       change_colors=False):
    '''
    matrix should be a numpy array
    '''
    fig, ax = plt.subplots(figsize=(9, 8))
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
            cell_value = matrix[i, j]
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


def plot_tSNE(tsne_output, head_scores_info):
    colours = ['red', 'green', 'blue', 'black']  # mark tasks with colours
    markers = ['^', 'o', '*', 'X', 's']  # mark languages with plus, circle, *, X, square

    dict_task_colors = {task: colours[i] for i, task in enumerate(ALLOWED_DATASETS)}
    dict_language_markers = {lang: markers[i] for i, lang in enumerate(LANGUAGES)}

    plt.figure(figsize=(8, 4))

    for point, (task, language, _) in zip(tsne_output, head_scores_info):
        plt.scatter(point[0],
                    point[1],
                    color=dict_task_colors[task],
                    marker=dict_language_markers[language])

    task_legend_handles = []
    for task in ALLOWED_DATASETS:
        task_legend_handles.append(
            plt.Line2D([], [],
                       color=dict_task_colors[task],
                       marker="_",
                       linestyle='None',
                       markeredgewidth=3.,
                       markersize=10))

    language_legend_handles = []
    for lang in LANGUAGES:
        language_legend_handles.append(
            plt.Line2D([], [],
                       color='gray',
                       marker=dict_language_markers[lang],
                       markeredgewidth=3.,
                       linestyle='None',
                       markersize=10))

    plt.gca().add_artist(
        plt.legend(task_legend_handles,
                   ALLOWED_DATASETS,
                   loc='best',
                   title='Tasks',
                   fontsize=12,
                   bbox_to_anchor=(0.99, 0.5)))

    plt.legend(language_legend_handles,
               LANGUAGES,
               loc='lower left',
               title='Languages',
               fontsize=12,
               bbox_to_anchor=(1.02, 0.35))

    plt.xlabel("Dimension 1", fontsize=14)
    plt.ylabel("Dimension 2", fontsize=14)
    plt.title("t-SNE visualization of subnetworks' masks", fontsize=18)
    plt.tight_layout()
    plt.grid()
    path = os.path.join('results','plots','tsne.pdf')
    plt.savefig(path)
    plt.show()