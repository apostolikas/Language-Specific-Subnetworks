import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

import argparse
import torch
import pandas as pd
from tqdm import trange, tqdm
from pathlib import Path
from transformers import AutoTokenizer, DataCollatorForTokenClassification, DataCollatorWithPadding

from data import ALLOWED_LANGUAGES, get_dataset, ALLOWED_DATASETS, WIKIANN_NAME
from mask import set_seed
from utils import StitchNet

from mask import get_dataloader, set_seed, get_dataset
from eval import get_model_accuracy, get_model_f1
from utils.affine_stitch import init_XML


def randomize_mask(mask):
    mask = mask.T
    for i, row in enumerate(mask):
        mask[i] = row[torch.randperm(len(row))]
    return mask.T


def valid(dataset, checkpoint):
    model = Path(checkpoint).parent.stem
    n_classes1 = get_dataset(dataset, tokenizer=None, no_load=True).n_classes
    n_classes2 = get_dataset(model, tokenizer=None, no_load=True).n_classes
    return n_classes1 == n_classes2


def stitch(args):

    if not valid(args.dataset, args.checkpoint2):
        raise ValueError("End model not compatible with this dataset")

    set_seed(args.seed)

    # Data
    tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base', use_fast=True)

    if args.dataset == WIKIANN_NAME:
        collate_fn = DataCollatorForTokenClassification(tokenizer=tokenizer)  #pad_to_multiple_of=8)
    else:
        collate_fn = DataCollatorWithPadding(tokenizer)

    label_names = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']
    id2label = {i: label for i, label in enumerate(label_names)}
    data_loader = get_dataloader(args, args.dataset, tokenizer, args.lang, collate_fn)

    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = StitchNet(args.checkpoint1, args.checkpoint2, args.layer, id2label).to(device)
    model.find_optimal_init(data_loader)
    model.load_masks(args.mask1, args.mask2)

    # Shuffle mask of first net
    if args.randomize:
        model.front_mask = randomize_mask(model.front_mask)

    # Remove all heads of first net
    if args.remove:
        model.front_mask *= 0
    if args.remove_end:
        model.end_mask *= 0

    optimizer = torch.optim.Adam(model.transform.parameters(), lr=args.lr)

    # Train
    print("Train stitching..")
    n_iters = len(data_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_iters, eta_min=1e-5)
    for epoch in range(args.epochs):
        mean_loss = 0
        for batch in tqdm(data_loader):
            batch = {k: v.to(device, non_blocking=True) for (k, v) in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch)[0]
            loss.backward()
            mean_loss += loss.item() / len(data_loader)
            optimizer.step()
            scheduler.step()

        print(f"Epoch {epoch+1}/{args.epochs} loss: {mean_loss:.3f} last_loss: {loss.item():.3f}")

    print('Done.')

    # Eval
    model_name1 = Path(args.checkpoint1).parent.stem
    model_name2 = Path(args.checkpoint2).parent.stem
    mask_task1 = Path(args.mask1).parent.stem
    mask_task2 = Path(args.mask2).parent.stem
    mask_lang1, mask_seed1 = os.path.split(args.mask1)[-1].split('.')[0].split('_')
    mask_lang2, mask_seed2 = os.path.split(args.mask2)[-1].split('.')[0].split('_')
    dataset = get_dataset(args.dataset, tokenizer, args.lang, split="test")
    baseline_ckp = args.checkpoint1.replace(model_name1, args.dataset)

    baseline = init_XML(baseline_ckp, id2label)
    baseline.to(device)

    baseline_mask = os.path.join(args.mask_dir, args.dataset, f"{args.lang}_0.pkl")
    baseline_mask = torch.load(baseline_mask)

    # Just the metric name and the function to get the metric results changes
    if args.dataset == WIKIANN_NAME:
        get_metric_results = get_model_f1
        # metric_name = 'f1'
    else:  # sequence classification
        get_metric_results = get_model_accuracy
        # metric_name = 'acc'

    results = {
        "dataset": args.dataset,
        "lang": args.lang,
        "layer": args.layer,
        "front_model": model_name1,
        "end_model": model_name2,
        "front_mask": mask_task1,
        "end_mask": mask_task2,
        "front_lang": mask_lang1,
        "end_lang": mask_lang2,
        "front_seed": mask_seed1,
        "end_seed": mask_seed2,
        f"baseline_acc": get_metric_results(baseline, dataset, 32, baseline_mask),
        f"stitch_acc": get_metric_results(model, dataset, 32)
    }
    model.remove_hooks()

    # Front model's output might have mismatch
    if valid(args.dataset, args.checkpoint1):
        front_results = get_metric_results(model.front_model, dataset, 32, model.front_mask)
    else:
        front_results = -1

        results.update({
            f"front_acc": front_results,
            f"end_acc": get_metric_results(model.end_model, dataset, 32, model.end_mask),
        })
        print(results)

    # Save
    save_dir = os.path.split(args.save_path)[0]
    os.makedirs(save_dir, exist_ok=True)
    df = pd.DataFrame.from_dict({k: [v] for (k, v) in results.items()})
    df.to_csv(args.save_path, mode='a', header=not os.path.exists(args.save_path), index=False)
    print("Saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simple settings.')
    parser.add_argument('checkpoint1', type=str)
    parser.add_argument('checkpoint2', type=str)
    parser.add_argument('mask1', type=str)
    parser.add_argument('mask2', type=str)
    parser.add_argument('layer', type=int)
    parser.add_argument('dataset', type=str, choices=ALLOWED_DATASETS)
    parser.add_argument('lang', type=str, choices=ALLOWED_LANGUAGES)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--sample-n', type=int, default=10000)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--mask-dir', type=str, default='results/pruned_masks')
    parser.add_argument('--save-path', type=str, default='results/stitch/dev.csv')
    parser.add_argument('--randomize', action='store_true')
    parser.add_argument('--remove', action='store_true')
    parser.add_argument('--remove-end', action='store_true')
    args = parser.parse_args()
    stitch(args)