import argparse
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BatchEncoding
import random
import numpy as np
import torch
import os

from bert_experiments import mask_model, prune_model

parser = argparse.ArgumentParser(description='Simple settings.')
parser.add_argument('checkpoint', type=str)
parser.add_argument('task', type=str, default='marco')
parser.add_argument('seed', type=int, default=0)
parser.add_argument('lang', type=str, choices=['en', 'de', 'fr', 'es', 'zh'])
parser.add_argument('output_dir', type=str, default='results')


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def tokenize_data(dataset: Dataset) -> BatchEncoding:
    encodings = ...
    return encodings


def get_dataloader(dataset_name: str, lang: str, batch_size: int = 32):
    dataset = load_dataset(dataset_name, lang)['validation']
    dataset = list(map(tokenize_data, dataset))
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset,
                            sampler=sampler,
                            batch_size=batch_size,
                            pin_memory=True,
                            num_workers=4)
    # TODO: is dataloader viable here?
    # TODO: Batches must match with format: input_ids, input_mask, segment_ids, label_ids = batch
    return dataloader


def main(args):

    # Arguments based on class
    dataset_name = ...  # This is determined based on the task
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data
    dataloader = get_dataloader(dataset_name, args.lang)

    # Model
    tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base', use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint).to(device)

    # Pruning
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    head_mask, mlp_mask = mask_model(model, dataloader, device)
    model = prune_model(args, model, dataloader, device, head_mask, mlp_mask)

    # Save
    pass


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)