import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

import argparse
import torch
from pathlib import Path
import numpy as np
from transformers import (AutoTokenizer,
                          AutoModelForSequenceClassification,
                          TrainingArguments,
                          DataCollatorWithPadding,
                          Trainer)
import evaluate

from data import get_dataset, ALLOWED_DATASETS, ALLOWED_LANGUAGES


def compute_metrics():

    metric = evaluate.load('accuracy')

    def accuracy(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return metric.compute(predictions=predictions, references=labels)

    return accuracy


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

    # Define training hyperparameters
    training_args = TrainingArguments(output_dir="./",
                                      report_to=None,
                                      disable_tqdm=True,
                                      per_device_eval_batch_size=args.batch_size,
                                      dataloader_num_workers=2,
                                      fp16=True)
    trainer = Trainer(model,
                      training_args,
                      data_collator=DataCollatorWithPadding(tokenizer),
                      tokenizer=tokenizer,
                      compute_metrics=compute_metrics())

    # Eval
    results = {}
    for lang, dataset in datasets.items():
        results[lang] = trainer.evaluate(dataset)['eval_accuracy']

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

    args = parser.parse_args()

    main(args)
