import argparse
import glob
import itertools
import json
import os
import sys
from pathlib import Path

from data import ALLOWED_DATASETS, ALLOWED_LANGUAGES

CONFIGS = {
    "stitch_across_tasks_and_languages": {
        "model_name1": [f"results/models/{model}/best" for model in ALLOWED_DATASETS],
        "model_name2": [f"results/models/{model}/best" for model in ALLOWED_DATASETS],
        "lang": ALLOWED_LANGUAGES,
        "layer": [i for i in range(12)]
    },
    "stitch_across_seeds": {
        "model_name": [f"results/models/{model}/best" for model in ALLOWED_DATASETS],
        "lang": ALLOWED_LANGUAGES,
        "layer": [i for i in range(12)],
        "seed1": [i for i in range(5)],
        "seed2": [i for i in range(5)],
    },
}


def stitch_across_tasks_and_languages(model_name1, model_name2, lang, layer):
    dataset = Path(model_name2).parent.stem
    mask1 = f"results/pruned_masks/{dataset}/{lang}_0.pkl"
    mask2 = f"results/pruned_masks/{dataset}/{lang}_0.pkl"
    return f"python stitch.py {model_name1} {model_name2} {mask1} {mask2} {layer} {dataset} {lang} --save-path results/stitch/stitch_across_tasks.csv"


def stitch_across_seeds(model_name, lang, layer, seed1, seed2):
    if seed2 != (seed1 + 1) % 5:
        return None
    dataset = Path(model_name).parent.stem
    mask1 = f"results/pruned_masks/{dataset}/{lang}_{seed1}.pkl"
    mask2 = f"results/pruned_masks/{dataset}/{lang}_{seed2}.pkl"
    return f"python stitch.py {model_name} {model_name} {mask1} {mask2} {layer} {dataset} {lang} --save-path results/stitch/stitch_across_seeds.csv"


def main(args):

    options = CONFIGS[args.config]
    keys, values = zip(*options.items())
    tasks = [dict(zip(keys, v)) for v in itertools.product(*values)]
    tasks = [globals()[args.config](**inputs) for inputs in tasks]
    tasks = [t for t in tasks if t is not None]

    os.makedirs(args.out_dir, exist_ok=True)
    tasks_file = os.path.join(args.out_dir, f"{args.config}.tasks")
    with open(tasks_file, 'w') as f:
        for task in tasks:
            f.write(f"{task}\n")
    print(f"{len(tasks)} tasks generated to {tasks_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simple settings.')
    parser.add_argument('config', help="one key from CONFIGS dictionary")
    parser.add_argument('--out-dir', default='tasks/')
    args = parser.parse_args()
    main(args)