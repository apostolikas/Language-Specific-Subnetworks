import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

import argparse
from torch.utils.data import DataLoader
import torch
from tqdm import trange

from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding

from data import ALLOWED_LANGUAGES, get_dataset, ALLOWED_DATASETS
from mask import set_seed
from utils import get_activations, cka


def get_dataloader(args, dataset_name, tokenizer, lang, **kwargs):
    dataset = get_dataset(dataset_name,
                          tokenizer,
                          lang,
                          "train",
                          args.sample_n,
                          tokenizer_kwargs={"max_length": 128})
    data_loader = DataLoader(dataset,
                             batch_size=args.batch_size,
                             pin_memory=True,
                             num_workers=4,
                             drop_last=True,
                             collate_fn=DataCollatorWithPadding(tokenizer, **kwargs))

    return data_loader


def cka_compare(args):

    # Common arguments, shared between different languages
    tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base', use_fast=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_paths = [args.checkpoint1, args.checkpoint2]
    mask_paths = [args.mask1, args.mask2]

    activations = []

    print("Calculating activations..")
    for model_path, mask_path in zip(model_paths, mask_paths):

        dataset_name = Path(model_path).parent.stem
        lang = os.path.split(mask_path)[-1].split('_')[0]

        # In order to match activations, we will need the exact same batch order
        # For this reason, we need a fixed seed
        set_seed(0)

        # Define data pipeline and pretrained model
        train_loader = get_dataloader(args,
                                      dataset_name,
                                      tokenizer=tokenizer,
                                      lang=lang,
                                      padding='max_length',
                                      max_length=128)
        model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)

        # Get masks
        head_mask = torch.load(mask_path)

        # Get activations
        act = get_activations(model, train_loader, head_mask=head_mask)
        activations.append(act)

    # Retrieve number and layers
    n_layers = model.config.num_hidden_layers

    # Calculate CKA match for each layer, each head
    cka_results = torch.zeros((n_layers, n_layers), dtype=torch.float32)
    for i in trange(n_layers, desc='Calculating CKA..'):
        for j in range(n_layers):
            a = activations[0][i].clone().to(device, non_blocking=True)
            b = activations[1][j].clone().to(device, non_blocking=True)
            cka_results[i, j] = cka(a, b)

    # Save
    ckp1 = Path(args.checkpoint1).parent.stem
    ckp2 = Path(args.checkpoint1).parent.stem
    mask1 = os.path.split(args.mask1)[-1].split('.')[0]
    mask2 = os.path.split(args.mask2)[-1].split('.')[0]
    filename = f"{ckp1}_{mask1}_{ckp2}_{mask2}.pkl"
    save_path = os.path.join(args.save_dir, filename)
    os.makedirs(args.save_dir, exist_ok=True)
    torch.save(cka_results, save_path)
    print("Saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simple settings.')
    parser.add_argument('checkpoint1', type=str)
    parser.add_argument('checkpoint2', type=str)
    parser.add_argument('mask1', type=str)
    parser.add_argument('mask2', type=str)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--sample-n', type=int, default=8192)
    parser.add_argument('--mask-dir', type=str, default='results/pruned_masks')
    parser.add_argument('--save-dir', type=str, default='results/cka')

    args = parser.parse_args()
    cka_compare(args)