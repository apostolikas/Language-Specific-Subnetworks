import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

import argparse
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from datasets import load_dataset
import random
import numpy as np
import torch

from pathlib import Path
from transformers import (AutoTokenizer,
                          AutoModelForSequenceClassification,
                          TrainingArguments,
                          DataCollatorWithPadding,
                          Trainer)
import evaluate

from data import get_dataset, ALLOWED_DATASETS, ALLOWED_LANGUAGES

from bert_experiments import mask_heads


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_dataloader(args, dataset_name, tokenizer, lang):
    dataset = get_dataset(dataset_name, tokenizer, lang, "train", args.sample_n)
    data_loader = DataLoader(dataset,
                             batch_size=args.batch_size,
                             pin_memory=True,
                             num_workers=1,
                             collate_fn=DataCollatorWithPadding(tokenizer))

    return data_loader


def main(args):

    # Common arguments, shared between different languages
    dataset_name = Path(args.checkpoint).parent.stem
    tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base', use_fast=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for lang in args.langs:

        set_seed(args.seed)

        # Define data pipeline and pretrained model
        train_loader = get_dataloader(args, dataset_name, tokenizer=tokenizer, lang=lang)
        model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint).to(device)

        # Calculate mask
        head_mask = mask_heads(args, model, train_loader, dataset_name)

        # Save
        print(head_mask)


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
    parser.add_argument('--sample-n', type=int, default=100)
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