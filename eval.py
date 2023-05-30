import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

import argparse
import torch
from pathlib import Path
import numpy as np
from tqdm import tqdm
from finetune import compute_ner_metrics
from torch.utils.data import DataLoader
from transformers import (AutoTokenizer,
                          AutoModelForSequenceClassification,
                          TrainingArguments,
                          DataCollatorWithPadding,
                          Trainer,
                          DataCollatorForTokenClassification)
import evaluate

from data import get_dataset, ALLOWED_DATASETS, ALLOWED_LANGUAGES


def compute_metrics():

    metric = evaluate.load('accuracy')

    def accuracy(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return metric.compute(predictions=predictions, references=labels)

    return accuracy


def get_model_accuracy(model, dataset, batch_size, head_mask=None):

    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             pin_memory=True,
                             num_workers=4,
                             drop_last=False,
                             collate_fn=DataCollatorWithPadding(dataset.tokenizer))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    total, hits = 0, 0

    if head_mask is not None:
        head_mask = head_mask.to(device)

    for batch in tqdm(data_loader, desc='Evaluation'):
        batch = {k: v.to(device, non_blocking=True) for (k, v) in batch.items()}
        if head_mask is not None:
            batch.update({"head_mask": head_mask})
        with torch.no_grad():
            logits = model(**batch)[1]
            preds = logits.detach().argmax(dim=1)
            labels = batch['labels'].detach()
            total += len(preds)
            hits += (preds == labels).sum().item()

    return hits / total

def get_model_f1(model, dataset, batch_size, head_mask=None):
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             pin_memory=True,
                             num_workers=4,
                             drop_last=False,
                             collate_fn=DataCollatorForTokenClassification(dataset.tokenizer))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   
    if head_mask is not None:
        head_mask = head_mask.to(device)

    all_preds = []
    all_labels = []
    for batch in tqdm(data_loader, desc='Evaluation'):
        batch = {k: v.to(device, non_blocking=True) for (k, v) in batch.items()}
        if head_mask is not None:
            batch.update({"head_mask": head_mask})
        with torch.no_grad():
            logits = model(**batch)[1]
            preds = logits.detach().argmax(dim=-1)
            all_preds.extend(preds.cpu().tolist()) # maybe dummy way to do it
            all_labels.extend(batch['labels'].detach().cpu().tolist())
     
    f1 = compute_ner_metrics(all_labels, all_preds)
    return f1

def main(args):

    # Retreiving dataset name
    dataset_name = Path(args.checkpoint).parent.stem

    # Get datasets
    tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base', use_fast=True)
    datasets = {}
    for lang in args.langs:
        datasets[lang] = get_dataset(dataset_name, tokenizer=tokenizer, split='test', lang=lang)

    # Get model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint).to(device)

    # Eval
    results = {}
    for lang, dataset in datasets.items():
        if args.mask:
            head_mask = torch.load(
                os.path.join(args.mask_dir, dataset_name, f"{lang}_{args.mask_seed}.pkl"))
        else:
            head_mask = None
        results[lang] = get_model_accuracy(model, dataset, args.batch_size, head_mask)

    for lang, acc in results.items():
        print(f"Results for {lang}: {acc * 100:.2f}%")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Simple settings.')
    parser.add_argument('checkpoint', type=str, help="path to checkpoint")
    parser.add_argument('--langs',
                        type=str,
                        nargs="+",
                        choices=ALLOWED_LANGUAGES,
                        default=ALLOWED_LANGUAGES)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--mask', action='store_true')
    parser.add_argument('--mask-dir', default='results/pruned_masks')
    parser.add_argument('--mask-seed', type=int, default=0)

    args = parser.parse_args()

    main(args)
