import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

import argparse
import torch
import pandas as pd
from tqdm import trange, tqdm

from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from data import ALLOWED_LANGUAGES, get_dataset, ALLOWED_DATASETS
from mask import set_seed
from utils import StitchNet

from mask import get_dataloader, set_seed
from eval import get_model_accuracy


def stitch(args):

    set_seed(args.seed)

    # data
    tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base', use_fast=True)
    data_loader = get_dataloader(args, args.dataset, tokenizer, args.lang)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = StitchNet(args.checkpoint1, args.checkpoint2, args.layer).to(device)
    model.find_optimal_init(data_loader)
    model.load_masks(args.mask1, args.mask2)

    optimizer = torch.optim.Adam(model.transform.parameters(), lr=args.lr)

    # Train
    print("Train stitching..")
    n_iters = len(data_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_iters)
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

        print(f"Epoch {epoch+1}/{args.epochs} loss: {mean_loss:.3f}")

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
    baseline = AutoModelForSequenceClassification.from_pretrained(baseline_ckp).to(device)
    baseline_mask = os.path.join(args.mask_dir, args.dataset, f"{args.lang}_0.pkl")
    baseline_mask = torch.load(baseline_mask)

    results = {
        "dataset": args.dataset,
        "lang": args.lang,
        "front_model": model_name1,
        "end_model": model_name2,
        "front_mask": mask_task1,
        "end_mask": mask_task2,
        "front_lang": mask_lang1,
        "end_lang": mask_lang2,
        "front_seed": mask_seed1,
        "end_seed": mask_seed2,
        "baseline_acc": get_model_accuracy(baseline, dataset, 32, baseline_mask),
        "stitch_acc": get_model_accuracy(model, dataset, 32)
    }
    model.remove_hooks()
    results.update({
        f"front_acc": get_model_accuracy(model.front_model, dataset, 32, model.front_mask),
        f"end_acc": get_model_accuracy(model.end_model, dataset, 32, model.end_mask),
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
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--mask-dir', type=str, default='results/pruned_masks')
    parser.add_argument('--save-path', type=str, default='results/stitch/dev.csv')

    args = parser.parse_args()
    stitch(args)