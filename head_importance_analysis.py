'''
This is just a dummy script to see how the importance scores are saved
'''
import pickle
import os
import torch

task = 'xnli'
lang='en'
seed=0
path = os.path.join('./results/importance_scores', task, f'head_imp_en_3.pickle')
#inf value when head_importance = 0 i.e. it is already removes
with open(path,'rb') as file:
    importance_scores = pickle.load(file)
    #! they are in [0,1] interval so they are a bit useless these statistics
    statistics = [] 
    for i in range(len(importance_scores)):
        tmp_importance_scores = importance_scores[i].flatten()
        mask = torch.isinf(tmp_importance_scores)

    # Use the mask to extract non-infinity elements
        non_infinity_elements = tmp_importance_scores[~mask]   
        mean = torch.mean(non_infinity_elements).item()
        std = torch.std(non_infinity_elements).item()
        mean = round(mean,3)
        std = round(std,3)
        statistics.append(f'{mean}\u00B1{std}')
#todo add plots for same language, same task, same seed and see how the head importance scores change with time