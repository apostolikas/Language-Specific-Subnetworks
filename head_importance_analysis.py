import pickle
import os
import torch
import cv2
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import pairwise_distances
from sklearn.manifold import TSNE
from collections import defaultdict
from mask import set_seed
import argparse
from make_plots import plot_square_matrix, create_big_plot, plot_tSNE
from call_plots import show_timesteps_head_scores
import matplotlib.pyplot as plt
from data import ALLOWED_DATASETS

# settings
LANGUAGES = ['en','de','fr','es','zh']

NUM_SEEDS = 5
SEEDS = [i for i in range(NUM_SEEDS)]

#I get a memory leak for k-means without this line
os.environ["OMP_NUM_THREADS"] = '1'
def load_head_importance_scores():
    '''
    Return:
        last_step_head_scores: A list of lists containing the 1d flattened head importance scores. This is the input to the clustering.
        last_step_head_scores_info: A list of lists containing tuples of (task, language, seed). This is related info to interpret the clustering output.
        all_steps_head_scores: A list of lists of 2d tensors containing the head importance scores at each timestep. Useful for timestep plots
    '''
    last_step_head_scores = [] 
    all_steps_head_scores = []
    last_step_head_scores_info = []  
    max_steps = -1
    for lang in LANGUAGES:
        for task in ALLOWED_DATASETS:
            for seed in SEEDS:
                path = os.path.join('./results/pruned_masks', task, f'head_imp_{lang}_{seed}.pickle')
                with open(path,'rb') as file:
                    importance_scores = pickle.load(file)
                    #inf value when head_importance is already removed so I put 0 there
                    tmp_all_head_scores = [torch.where(tmp_importance_scores == float('inf'),0, tmp_importance_scores) 
                                    for tmp_importance_scores in importance_scores] 
                    tmp_last_head_scores = tmp_all_head_scores[-1].flatten().tolist()

                    last_step_head_scores.append(tmp_last_head_scores) 
                    last_step_head_scores_info.append((task, lang, seed))
                    if max_steps < len(tmp_all_head_scores):
                        max_steps = len(tmp_all_head_scores)
                    all_steps_head_scores.append(tmp_all_head_scores)

    # assert(len(last_step_head_scores) == len(last_step_head_scores_info) == len(all_steps_head_scores) == 75)
    print(f'max time steps{max_steps}')

    return last_step_head_scores, last_step_head_scores_info, all_steps_head_scores

def hierarchical_clustering(dist_matrix, num_clusters):
    clustering = AgglomerativeClustering(n_clusters=num_clusters, linkage='complete')
    cluster_labels = clustering.fit_predict(dist_matrix)
    return cluster_labels

def kmeans_clustering(dist_matrix, num_clusters):
    seed = 0
    set_seed(seed) # maybe not needed
    kmeans = KMeans(n_clusters=num_clusters, random_state=seed, n_init=30) #default 10 random init, the more the better
    cluster_labels = kmeans.fit_predict(dist_matrix)
    return cluster_labels

def cluster_statistics(last_step_head_scores_info, cluster_labels, num_clusters):
    '''
    Input:
        last_step_head_scores_info: A list of lists containing tuples of (task, language, seed). This is related info to interpret the clustering output.
        cluster_labels: A list containing the assigned label for each input
        num_clusters: int how many clusters we used
    '''
    cluster_languages = defaultdict(list) # {cluster_id : list of languages in this cluster_id}
    cluster_tasks = defaultdict(list) # {cluster_id : list of tasks in this cluster_id}
    cluster_seeds = defaultdict(list) # {cluster_id : list of seeds in this cluster id}
    for i, (task, lang, seed) in enumerate(last_step_head_scores_info):
        tmp_cluster_label = cluster_labels[i]
        cluster_seeds[tmp_cluster_label].append(seed)
        cluster_languages[tmp_cluster_label].append(lang)
        cluster_tasks[tmp_cluster_label].append(task)

    for tmp_cluster_label in range(num_clusters):
        tmp_seeds = set(cluster_seeds[tmp_cluster_label])
        tmp_languages = set(cluster_languages[tmp_cluster_label])
        tmp_tasks = set(cluster_tasks[tmp_cluster_label])
        print('-'*20)
        print(f"Cluster {tmp_cluster_label} Statistics:")
        print(f"Languages: {', '.join(tmp_languages)}")
        print(f"Tasks: {', '.join(tmp_tasks)}")
        print(f"Seeds: {tmp_seeds}")

def apply_tSNE(input):
    # Create a t-SNE object
    tsne = TSNE(n_components=2, random_state=42, perplexity=10, n_iter=1500)
    output = tsne.fit_transform(input)
    print('KL divergence ', tsne.kl_divergence_)
    return output

def main(args):
    last_step_head_scores, last_step_head_scores_info, all_steps_head_scores = load_head_importance_scores()
    array_last_step_head_scores = np.array(last_step_head_scores)
    #t-SNE
    tsne_output = apply_tSNE(array_last_step_head_scores)
    plot_tSNE(tsne_output, last_step_head_scores_info)
    
    # Compute the distance matrix
    dist_matrix = pairwise_distances(last_step_head_scores, metric=args.distance_metric)

    if args.algorithm == 'kmeans':
        cluster_labels = kmeans_clustering(dist_matrix, args.num_clusters)
    elif args.algorithm == 'hierarchical':
        cluster_labels = hierarchical_clustering(dist_matrix, args.num_clusters)
    else:
        raise NotImplementedError("We only support kmeans or hierarchical")

    cluster_statistics(last_step_head_scores_info, cluster_labels, args.num_clusters)

    if args.plot_timesteps:
        show_timesteps_head_scores(all_steps_head_scores, last_step_head_scores_info)
   
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Head importance analysis')
    parser.add_argument('--num_clusters', type=int, default=3)
    parser.add_argument('--algorithm', default='kmeans', choices=['kmeans', 'hierarchical'], 
                        help='clustering algorithm that will be used either kmeans or hierarchical')
    parser.add_argument('--plot_timesteps',type=bool, default=False) # a lot of big plots 
    parser.add_argument('--distance_metric', type=str,choices=['euclidean', 'cosine'], default='cosine' )
    args = parser.parse_args()
    main(args)