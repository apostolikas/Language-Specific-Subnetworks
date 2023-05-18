import matplotlib.pyplot as plt
import numpy as np
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

def plot_square_matrix(matrix, x_labels, y_labels, save_path, x_title, y_title, title, colour_map=plt.cm.OrRd):
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
            ax.text(j, i, f"{cell_value:.2f}\n", va='center', ha='center')
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    # maybe we can remove also the titles I am not that it is a good idea
    # plt.title(title)
    plt.show()