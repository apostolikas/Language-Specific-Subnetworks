# Investigating the cross-lingual sharing mechanism of multilingual models through their subnetworks

![stitching_results](https://github.com/apostolikas/Language-Specific-Subnetworks/assets/9435563/48624967-ee28-49dd-a80e-bccfdb124cb4)

> It has been shown that different, equally good subnetworks exist in a transformer model after fine-tuning and pruning it [[1]](#prasanna2020bert). In this project, we investigate the similarity of unilingual subnetworks, obtained by structured-pruned multilingual transformer models. By comparing subnetworks based on (i) mask similarity, (ii) representation similarity, and (iii) functional similarity, we demonstrate that unilingual subnetworks can effectively solve the same task in different languages and solve other tasks in early layers, even with shuffled masks. However, the last layers of the subnetworks are task-specific and cannot generalize to other tasks. Our research also provides insight into mutual information shared between cross-lingual subnetworks.


Nikolaos Apostolikas, Gergely Papp, Panagiotis Tsakas, Vasileios Vythoulkas <br />
*March 2023*

-------------------

This is the official repository of **[Investigating the cross-lingual sharing mechanism of multilingual models through their subnetworks](#)**.
Please find instructions to reproduce the results below.

&nbsp;

## Preparation

### Download the repository
```
Download this repository.
cd Language-Specific-Subnetworks
```

### Install environment
Conda:
```
conda env create -f env.yml
conda activate atcs
```
Pip:
```
pip install -r requirements.txt
```

### Download finetuned models and masks
Either run our provided downloading script:
```
source download.sh
```
Or download manually by
 - Downloading the models from [google drive](https://drive.google.com/file/d/14xYRVCJbhxhkGR85JzizXn0Me-mMgEKa/view?usp=sharing)
 - Running `unzip results.zip`

### Experiments

#### Jaccard
```
python plot/load_masks.py
```
#### CKA
Syntax:`python cka.py model1 model2 mask1 mask2` <br />
Example:
```
python cka.py results/models/marc/best results/models/paws-x/best results/pruned_masks/marc/zh_0.pkl results/pruned_masks/paws-x/zh_0.pkl
```
This script saves results under results/cka folder.

#### Stitching
Syntax:`python stitch.py model1 model2 mask1 mask2 layer_index target_dataset target_lang` <br />
Example:
```
python stitch.py results/models/marc/best results/models/marc/best results/pruned_masks/marc/en_0.pkl results/pruned_masks/marc/en_0.pkl 6 marc en
```
This script saves results under results/stitch folder, in a csv.

### Plotting
Some of the plots can be found under `plot/notebooks/`. <br />
The t-SNE plot can be made via
```
python plot/head_importance_analysis.py
```
And the mask-overlap via
```
python plot/stats_script.py
```

### Finetuning + masking
In order to finetune a model on ROBERTa, run:
```
python finetune.py [xnli|paws-x|marc|wikiann]
```

Then, you can create the masks for the subnetworks with
```
python mask.py results/models/YOUR_MODEL/best --seed 0
```
This will script will save the 5 masks for 5 languages for the given finetuned model.

&nbsp;

<!-- ## Contributions

1. Jaccard similarity of masks is unstable across seeds and therefore it is not a reliable metric to compare subnetworks [[1]](#prasanna2020bert). In response to this, we present an analysis through representational and functional similarity metrics, namely CKA  [[2]](#cka) and relative accuracy (RA) achieved by model stitching [\[3,](#stitching1)[ 4\]](#stitching2). In contrast to Jaccard similarity, these measures are stable across seeds.

2. While subnetworks can be grouped into tasks and languages, we find that tasks have a greater impact on the final mask of subnetworks than languages. 

3. Linear CKA shows little or no relation between subnetworks that were trained for other tasks and languages. However, in fact, all subnetworks contain sufficient information to solve other tasks. More precisely, an affine stitching layer at early layers is enough to match any other subnetworks' performance, regardless of what task or language it was pruned for. Even more, masks in early layers can be shuffled without losing any information about the task, regardless of the language. -->

&nbsp;

## References

<a id="prasanna2020bert"></a> [1] Sai Prasanna, Anna Rogers, and Anna Rumshisky. 2020. When bert plays the lottery, all tickets are winning. arXiv preprint arXiv:2005.00561.

<a id="cka"></a> [2] Simon Kornblith, Mohammad Norouzi, Honglak Lee, and Geoffrey Hinton. 2019. Similarity of neural network representations revisited. In Proceedings of the 36th International Conference on Machine Learning, volume 97 of Proceedings of Machine Learning Research, pages 3519–3529. PMLR.

<a id="stitching1"></a> [3] Adrián Csiszárik, Péter K ̋orösi-Szabó, Ákos Matszangosz, Gergely Papp, and Dániel Varga. 2021. Similarity and matching of neural network representations. In Advances in Neural Information Processing Systems, volume 34, pages 5656–5668. Curran Associates, Inc.

<a id="stitching2"></a> [4] Yamini Bansal, Preetum Nakkiran, and Boaz Barak. 2021. Revisiting model stitching to compare neural representations. In Advances in Neural Information Processing Systems, volume 34, pages 225–236. Curran Associates, Inc.
