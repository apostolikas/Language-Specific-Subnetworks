# Investigating the cross-lingual sharing mechanism of multilingual models through their subnetworks

March 2023 | Nikolaos Apostolikas, Gergely Papp, Panagiotis Tsakas, Vasileios Vythoulkas

-------------------

&nbsp;

## Overview

It has been shown that different, equally good subnetworks exist in a transformer model after fine-tuning and pruning it [[1]](#prasanna2020bert). In this project, we investigate the similarity of unilingual subnetworks, obtained by structured-pruned multilingual transformer models. By comparing subnetworks based on (i) mask similarity, (ii) representation similarity, and (iii) functional similarity, we demonstrate that unilingual subnetworks can effectively solve the same task in different languages and solve other tasks in early layers, even with shuffled masks. However, the last layers of the subnetworks are task-specific and cannot generalize to other tasks. Our research also provides insight into mutual information shared between cross-lingual subnetworks.

&nbsp;

## Contributions

1. Jaccard similarity of masks is unstable across seeds and therefore it is not a reliable metric to compare subnetworks [[1]](#prasanna2020bert). In response to this, we present an analysis through representational and functional similarity metrics, namely CKA  [[2]](#cka) and relative accuracy (RA) achieved by model stitching [\[3,](#stitching1)[ 4\]](#stitching2). In contrast to Jaccard similarity, these measures are stable across seeds.

2. While subnetworks can be grouped into tasks and languages, we find that tasks have a greater impact on the final mask of subnetworks than languages. 

3. Linear CKA shows little or no relation between subnetworks that were trained for other tasks and languages. However, in fact, all subnetworks contain sufficient information to solve other tasks. More precisely, an affine stitching layer at early layers is enough to match any other subnetworks' performance, regardless of what task or language it was pruned for. Even more, masks in early layers can be shuffled without losing any information about the task, regardless of the language.

&nbsp;

## References

<a id="prasanna2020bert"></a> [1] Sai Prasanna, Anna Rogers, and Anna Rumshisky. 2020. When bert plays the lottery, all tickets are winning. arXiv preprint arXiv:2005.00561.

<a id="cka"></a> [2] Simon Kornblith, Mohammad Norouzi, Honglak Lee, and Geoffrey Hinton. 2019. Similarity of neural network representations revisited. In Proceedings of the 36th International Conference on Machine Learning, volume 97 of Proceedings of Machine Learning Research, pages 3519–3529. PMLR.

<a id="stitching1"></a> [3] Adrián Csiszárik, Péter K ̋orösi-Szabó, Ákos Matszangosz, Gergely Papp, and Dániel Varga. 2021. Similarity and matching of neural network representations. In Advances in Neural Information Processing Systems, volume 34, pages 5656–5668. Curran Associates, Inc.

<a id="stitching2"></a> [4] Yamini Bansal, Preetum Nakkiran, and Boaz Barak. 2021. Revisiting model stitching to compare neural representations. In Advances in Neural Information Processing Systems, volume 34, pages 225–236. Curran Associates, Inc.