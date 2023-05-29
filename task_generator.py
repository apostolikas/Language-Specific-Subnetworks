import argparse
import glob
import itertools
import json
import os
import sys
from pathlib import Path

from data import ALLOWED_DATASETS, ALLOWED_LANGUAGES

CONFIGS = {
    # STITCH
    "stitch_randomly": {
        "model_name": [f"results/models/{model}/best" for model in ALLOWED_DATASETS],
        "lang": ALLOWED_LANGUAGES,
        "layer": [i for i in range(12)]
    },
    "stitch_chaoticly": {
        "model_name1": [f"results/models/{model}/best" for model in ALLOWED_DATASETS],
        "model_name2": [f"results/models/{model}/best" for model in ALLOWED_DATASETS],
        "lang1": ALLOWED_LANGUAGES,
        "lang2": ALLOWED_LANGUAGES,
        "layer": [i for i in range(12)]
    },
    "stitch_across_languages": {
        "lang1": ALLOWED_LANGUAGES, "lang2": ALLOWED_LANGUAGES, "layer": [i for i in range(12)]
    },
    "stitch_across_tasks": {
        "model_name1": [f"results/models/{model}/best" for model in ALLOWED_DATASETS],
        "model_name2": [f"results/models/{model}/best" for model in ALLOWED_DATASETS],
        "layer": [i for i in range(12)]
    },
    "stitch_across_seeds": {
        "layer": [i for i in range(12)],
        "seed1": [i for i in range(5)],
        "seed2": [i for i in range(5)],
    },
    "stitch_validation": {
        "layer": [i for i in range(12)],
    },
    "stitch_removed": {
        "model_name": [f"results/models/{model}/best" for model in ALLOWED_DATASETS],
        "lang": ALLOWED_LANGUAGES,
        "layer": [i for i in range(12)],
    },
    "stitch_remove_last": {
        "model_name": [f"results/models/{model}/best" for model in ALLOWED_DATASETS],
        "lang": ALLOWED_LANGUAGES,
    },

  # CKA
    "cka_chaoticly": {
        "model_name1": [f"results/models/{model}/best" for model in ALLOWED_DATASETS],
        "model_name2": [f"results/models/{model}/best" for model in ALLOWED_DATASETS],
        "lang1": ALLOWED_LANGUAGES,
        "lang2": ALLOWED_LANGUAGES
    },
    "cka_across_languages": {
        "model_name": [f"results/models/{model}/best" for model in ALLOWED_DATASETS],
        "lang1": ALLOWED_LANGUAGES,
        "lang2": ALLOWED_LANGUAGES
    },
    "cka_across_tasks": {
        "model_name1": [f"results/models/{model}/best" for model in ALLOWED_DATASETS],
        "model_name2": [f"results/models/{model}/best" for model in ALLOWED_DATASETS],
        "lang": ALLOWED_LANGUAGES
    },
    "cka_across_seeds": {
        "model_name": [f"results/models/{model}/best" for model in ALLOWED_DATASETS],
        "lang": ALLOWED_LANGUAGES,
        "seed1": [i for i in range(5)],
        "seed2": [i for i in range(5)],
    },
    "cka_validation": {
        "model_name": [f"results/models/{model}/best" for model in ALLOWED_DATASETS],
        "lang": ALLOWED_LANGUAGES
    },
}


def stitch_randomly(model_name, lang, layer):
    dataset = Path(model_name).parent.stem
    mask = f"results/pruned_masks/{dataset}/{lang}_0.pkl"
    return f"python stitch.py {model_name} {model_name} {mask} {mask} {layer} {dataset} {lang} --save-path results/stitch/stitch_randomly.csv --randomize"


def stitch_chaoticly(model_name1, model_name2, lang1, lang2, layer):
    if model_name1 == model_name2 or lang1 == lang2:
        return
    dataset1 = Path(model_name1).parent.stem
    dataset2 = Path(model_name2).parent.stem
    mask1 = f"results/pruned_masks/{dataset1}/{lang1}_0.pkl"
    mask2 = f"results/pruned_masks/{dataset2}/{lang2}_0.pkl"
    return f"python stitch.py {model_name1} {model_name2} {mask1} {mask2} {layer} {dataset2} {lang2} --save-path results/stitch/stitch_chaotic.csv"


def stitch_across_languages(lang1, lang2, layer):
    if lang1 == lang2:
        return
    model_name = 'results/models/xnli/best'
    dataset = Path(model_name).parent.stem
    mask1 = f"results/pruned_masks/{dataset}/{lang1}_0.pkl"
    mask2 = f"results/pruned_masks/{dataset}/{lang2}_0.pkl"
    return f"python stitch.py {model_name} {model_name} {mask1} {mask2} {layer} {dataset} {lang2} --save-path results/stitch/stitch_across_languages.csv"


def stitch_across_tasks(model_name1, model_name2, layer):
    if model_name1 == model_name2:
        return
    lang = 'fr'
    dataset1 = Path(model_name1).parent.stem
    dataset2 = Path(model_name2).parent.stem
    mask1 = f"results/pruned_masks/{dataset1}/{lang}_0.pkl"
    mask2 = f"results/pruned_masks/{dataset2}/{lang}_0.pkl"
    return f"python stitch.py {model_name1} {model_name2} {mask1} {mask2} {layer} {dataset2} {lang} --save-path results/stitch/stitch_across_tasks.csv"


def stitch_across_seeds(layer, seed1, seed2):
    if seed2 != (seed1 + 1) % 5:
        return None
    model_name = 'results/models/xnli/best'
    lang = 'fr'
    dataset = Path(model_name).parent.stem
    mask1 = f"results/pruned_masks/{dataset}/{lang}_{seed1}.pkl"
    mask2 = f"results/pruned_masks/{dataset}/{lang}_{seed2}.pkl"
    return f"python stitch.py {model_name} {model_name} {mask1} {mask2} {layer} {dataset} {lang} --save-path results/stitch/stitch_across_seeds.csv"


def stitch_validation(layer):
    model_name = 'results/models/paws-x/best'
    lang = 'fr'
    dataset = Path(model_name).parent.stem
    mask = f"results/pruned_masks/{dataset}/{lang}_0.pkl"
    return f"python stitch.py {model_name} {model_name} {mask} {mask} {layer} {dataset} {lang} --save-path results/stitch/stitch_validation.csv"


def stitch_removed(model_name, lang, layer):
    dataset = Path(model_name).parent.stem
    mask1 = f"results/pruned_masks/{dataset}/{lang}_0.pkl"
    mask2 = f"results/pruned_masks/{dataset}/{lang}_0.pkl"
    return f"python stitch.py {model_name} {model_name} {mask1} {mask2} {layer} {dataset} {lang} --remove --save-path results/stitch/stitch_removed.csv"


def stitch_remove_last(model_name, lang):
    dataset = Path(model_name).parent.stem
    mask1 = f"results/pruned_masks/{dataset}/{lang}_0.pkl"
    mask2 = f"results/pruned_masks/{dataset}/{lang}_0.pkl"
    return f"python stitch.py {model_name} {model_name} {mask1} {mask2} 11 {dataset} {lang} --remove --remove-end --save-path results/stitch/stitch_remove_last.csv"


# =================================================================
# CKA
# =================================================================


def cka_chaoticly(model_name1, model_name2, lang1, lang2):
    if model_name1 == model_name2 or lang1 == lang2:
        return
    mask1 = f"results/pruned_masks/{Path(model_name1).parent.stem}/{lang1}_0.pkl"
    mask2 = f"results/pruned_masks/{Path(model_name2).parent.stem}/{lang2}_0.pkl"
    return f"python cka.py {model_name1} {model_name2} {mask1} {mask2} --save-dir results/cka/chaotic/"


def cka_across_languages(model_name, lang1, lang2):
    if lang1 == lang2:
        return
    mask1 = f"results/pruned_masks/{Path(model_name).parent.stem}/{lang1}_0.pkl"
    mask2 = f"results/pruned_masks/{Path(model_name).parent.stem}/{lang2}_0.pkl"
    return f"python cka.py {model_name} {model_name} {mask1} {mask2} --save-dir results/cka/across_langs/"


def cka_across_tasks(model_name1, model_name2, lang):
    if model_name1 == model_name2:
        return
    mask1 = f"results/pruned_masks/{Path(model_name1).parent.stem}/{lang}_0.pkl"
    mask2 = f"results/pruned_masks/{Path(model_name2).parent.stem}/{lang}_0.pkl"
    return f"python cka.py {model_name1} {model_name2} {mask1} {mask2} --save-dir results/cka/across_tasks/"


def cka_across_seeds(model_name, lang, seed1, seed2):
    if seed2 != (seed1 + 1) % 5:
        return None
    dataset = Path(model_name).parent.stem
    mask1 = f"results/pruned_masks/{dataset}/{lang}_{seed1}.pkl"
    mask2 = f"results/pruned_masks/{dataset}/{lang}_{seed2}.pkl"
    return f"python cka.py {model_name} {model_name} {mask1} {mask2} --save-dir results/cka/across_seeds/"


def cka_validation(model_name, lang):
    dataset = Path(model_name).parent.stem
    mask = f"results/pruned_masks/{dataset}/{lang}_0.pkl"
    return f"python cka.py {model_name} {model_name} {mask} {mask} --save-dir results/cka/validation/"


def main(args):

    for config in args.configs:

        options = CONFIGS[config]
        keys, values = zip(*options.items())
        tasks = [dict(zip(keys, v)) for v in itertools.product(*values)]
        tasks = [globals()[config](**inputs) for inputs in tasks]
        tasks = [t for t in tasks if t is not None]

        os.makedirs(args.out_dir, exist_ok=True)
        tasks_file = os.path.join(args.out_dir, f"{config}.tasks")
        with open(tasks_file, 'w') as f:
            for task in tasks:
                f.write(f"{task}\n")
        print(f"{config} - {len(tasks)} tasks generated to {tasks_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simple settings.')
    parser.add_argument('--configs', nargs="+", default=list(CONFIGS.keys()))
    parser.add_argument('--out-dir', default='tasks/')
    args = parser.parse_args()
    main(args)