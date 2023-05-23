import argparse
import torch
import os
import numpy as np
from transformers import (AutoTokenizer,
                          AutoModelForSequenceClassification,
                          TrainingArguments,
                          DataCollatorWithPadding, DataCollatorForTokenClassification,
                          Trainer,
                          EarlyStoppingCallback, AutoModelForTokenClassification)
import evaluate #we need !pip install seqeval

from data import get_dataset, ALLOWED_DATASETS


def compute_metrics():

    metric = evaluate.load('accuracy')

    def accuracy(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return metric.compute(predictions=predictions, references=labels)

    return accuracy

def get_ner_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    d= compute_ner_metrics(labels, predictions)
    return d

def compute_ner_metrics(labels, predictions):
    # Remove ignored index (special tokens) and convert to labels
    metric = evaluate.load("seqeval")
    label_names = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']
    true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
    f1 = all_metrics["overall_f1"] # micro-f1
    return {"f1":f1}

def main(args):

    # Get datasets
    tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base', use_fast=True)
    train = get_dataset(args.dataset, tokenizer=tokenizer, split='train', sample_n=args.sample_n)
    val = get_dataset(args.dataset, tokenizer=tokenizer, split='validation')
    test = get_dataset(args.dataset, tokenizer=tokenizer, split='test')

    # Get model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.dataset == 'wikiann':
        label_names = train.dataset.features["ner_tags"].feature.names
        id2label = {i: label for i, label in enumerate(label_names)}
        label2id = {v: k for k, v in id2label.items()}
        model = AutoModelForTokenClassification.from_pretrained(args.resume, id2label=id2label, label2id=label2id)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(args.resume,
                                                               num_labels=train.n_classes)
    model = model.to(device)

    # Define training hyperparameters
    training_args = TrainingArguments(output_dir=os.path.join(args.save_dir, args.dataset),
                                      seed=args.seed,
                                      evaluation_strategy="steps",
                                      save_strategy='steps',
                                      eval_steps=min(1000, args.sample_n),
                                      save_steps=min(1000, args.sample_n),
                                      learning_rate=args.lr,
                                      per_device_train_batch_size=args.batch_size,
                                      per_device_eval_batch_size=args.batch_size,
                                      num_train_epochs=args.epochs,
                                      weight_decay=args.weight_decay,
                                      load_best_model_at_end=True,
                                      metric_for_best_model="eval_loss",
                                      dataloader_num_workers=2,
                                      save_total_limit=1,
                                      fp16=True)
    if args.dataset == 'wikiann':
        trainer = Trainer(model,
                      training_args,
                      train_dataset=train,
                      eval_dataset=val,
                      data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer),
                      tokenizer=tokenizer,
                      compute_metrics=get_ner_metrics,
                      callbacks=[EarlyStoppingCallback(early_stopping_patience=3)])
    else:
        trainer = Trainer(model,
                        training_args,
                        train_dataset=train,
                        eval_dataset=val,
                        data_collator=DataCollatorWithPadding(tokenizer),
                        tokenizer=tokenizer,
                        compute_metrics=compute_metrics(),
                        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)])

    # Train
    print('Device ', device)
    trainer.train()

    # Eval
    print("Final dev evaluation")
    trainer.evaluate()
    print("Final test evaluation")
    trainer.evaluate(test)

    # Save
    save_path = os.path.join(args.save_dir, args.dataset, "best")
    print(f"Saving model to {save_path}")
    trainer.save_model(save_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Simple settings.')
    parser.add_argument('dataset', type=str, choices=ALLOWED_DATASETS)
    parser.add_argument('--resume', type=str, default="xlm-roberta-base", help="path to checkpoint")
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--weight-decay', type=float, default=1e-2)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save-dir', type=str, default="results/models")
    parser.add_argument('--sample-n',
                        type=int,
                        default=0,
                        help='Number of train samples. 0 means all.')

    args = parser.parse_args()

    main(args)
