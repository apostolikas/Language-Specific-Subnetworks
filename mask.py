import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

import argparse
from torch.utils.data import DataLoader
import random
import numpy as np
import torch
import pickle

from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, AutoModelForTokenClassification, DataCollatorForTokenClassification

from data import get_dataset, ALLOWED_LANGUAGES

from bert_experiments import mask_heads


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_dataloader(args, dataset_name, tokenizer, lang, collate_fn):
    dataset = get_dataset(dataset_name, tokenizer, lang, "train", args.sample_n)
    data_loader = DataLoader(dataset,
                             batch_size=args.batch_size,
                             pin_memory=True,
                             num_workers=4,
                             drop_last=True,
                             collate_fn=collate_fn)
                             #DataCollatorWithPadding(tokenizer))

    return data_loader

def main(args):

    # Common arguments, shared between different languages
    dataset_name = Path(args.checkpoint).parent.stem
    tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base', use_fast=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root = os.path.join(args.save_dir, dataset_name)

    for lang in args.langs:

        set_seed(args.seed)

        # Define data pipeline and pretrained model
        
        if dataset_name == 'wikiann':
            collate_fn = DataCollatorForTokenClassification(tokenizer=tokenizer)
            label_names = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']
            id2label = {i: label for i, label in enumerate(label_names)}
            label2id = {v: k for k, v in id2label.items()}
            model = AutoModelForTokenClassification.from_pretrained(args.checkpoint, id2label=id2label, label2id=label2id).to(device)
        else:
            collate_fn =DataCollatorWithPadding(tokenizer)
            model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint).to(device)

        train_loader =  get_dataloader(args, dataset_name, tokenizer=tokenizer, lang=lang, collate_fn =
                                             collate_fn)
        # Calculate mask
        head_mask, head_importance = mask_heads(args, model, train_loader, dataset_name)

        # Save head_mask
        os.makedirs(root, exist_ok=True)
        save_path = os.path.join(root, f"{lang}_{args.seed}.pkl")
        torch.save(head_mask, save_path)

        # Save head importance
        os.makedirs(root, exist_ok=True)
        save_path = os.path.join(root, f"head_imp_{lang}_{args.seed}.pickle")
        with open(save_path, 'wb') as f:
            pickle.dump(head_importance, f)

        print("Saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simple settings.')
    parser.add_argument('checkpoint', type=str)
    parser.add_argument('--langs',
                        type=str,
                        nargs="+",
                        choices=ALLOWED_LANGUAGES,
                        default=ALLOWED_LANGUAGES)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--sample-n', type=int, default=5000)
    parser.add_argument('--save-dir', type=str, default='results/pruned_masks')
    parser.add_argument('--masking-amount',
                        type=float,
                        default=0.1,
                        help="Amount to heads to masking at each masking step.")
    parser.add_argument(
        '--masking-threshold',
        type=float,
        default=0.9,
        help=
        "masking threshold in term of metrics (stop masking when metric < threshold * original metric value)."
    )
    parser.add_argument("--dont_normalize_importance_by_layer",
                        action="store_true",
                        help="Don't normalize importance score by layers")
    parser.add_argument(
        "--dont_normalize_global_importance",
        action="store_true",
        help="Don't normalize all importance scores between 0 and 1",
    )
    args = parser.parse_args()
    main(args)