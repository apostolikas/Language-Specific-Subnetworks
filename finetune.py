import argparse
import torch
import os
import numpy as np
import math
from typing import Optional
from transformers import (AutoTokenizer,
                          AutoModelForSequenceClassification,
                          TrainingArguments,
                          DataCollatorWithPadding, DataCollatorForTokenClassification,
                          Trainer,
                          EarlyStoppingCallback, AutoModelForTokenClassification,
                          XLMRobertaForMaskedLM, DataCollatorForLanguageModeling)
import evaluate #we need !pip install seqeval
from datasets import load_metric

from data import get_dataset, ALLOWED_DATASETS, WIKIANN_NAME, WIKIPEDIA_NAME
os.environ["WANDB_DISABLED"] = "true" # just for canvas submission
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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


class WikipediaTrainer(Trainer):
    def evaluate(self, eval_dataset=None, ignore_keys: Optional[list[str]] = None,
        metric_key_prefix: str = "eval"):
        if eval_dataset is None: # validation
            eval_dataset = self.eval_dataset
            split = 'validation'
        else:
            split = 'test'
            metric_key_prefix = 'test'
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        steps = 0
        total_loss = 0
        for batch in eval_dataloader:
            input_ids, input_mask, label_ids = batch['input_ids'], batch['attention_mask'], batch['labels']
            input_ids, input_mask, label_ids = input_ids.to(device), input_mask.to(device), label_ids.to(device)
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask = input_mask, labels = label_ids)
                total_loss += outputs.loss.detach().mean().item()
                steps += 1

        avg_loss = total_loss / steps
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        print(f'split {split} {metric_key_prefix} metric_key_prefix perplexity {perplexity} len {len(eval_dataloader)}')
        return {f"{metric_key_prefix}_perplexity": perplexity}

def main(args):
    # Get datasets
    tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base', use_fast=True)
    
    if args.dataset == WIKIPEDIA_NAME:
        tokenizer.pad_token = tokenizer.eos_token #! that's what they do here https://huggingface.co/docs/transformers/main/tasks/masked_language_modeling

    train = get_dataset(args.dataset, tokenizer=tokenizer, split='train', sample_n=args.sample_n)
    val = get_dataset(args.dataset, tokenizer=tokenizer, split='validation',sample_n=args.sample_n)
    test = get_dataset(args.dataset, tokenizer=tokenizer, split='test',sample_n=args.sample_n)

    # Get model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # default settings for non-wikipedia datasets
    metric_for_best_model = "eval_loss"
    greater_is_better = None 
    if args.dataset == WIKIANN_NAME:
        label_names = train.dataset.features["ner_tags"].feature.names
        id2label = {i: label for i, label in enumerate(label_names)}
        label2id = {v: k for k, v in id2label.items()}
        model = AutoModelForTokenClassification.from_pretrained(args.resume, id2label=id2label, label2id=label2id)
    elif args.dataset == WIKIPEDIA_NAME:
        # metric = load_metric("perplexity")
        model = XLMRobertaForMaskedLM.from_pretrained(args.resume)
        metric_for_best_model ='perplexity'
        greater_is_better = False
    else:
        model = AutoModelForSequenceClassification.from_pretrained(args.resume,
                                                               num_labels=train.n_classes)
    model = model.to(device)

     

    
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
                                    metric_for_best_model=metric_for_best_model,
                                    dataloader_num_workers=2,
                                    save_total_limit=1,
                                    fp16=True, greater_is_better = greater_is_better)
    if args.dataset == WIKIANN_NAME: #NER
        trainer = Trainer(model,
                      training_args,
                      train_dataset=train,
                      eval_dataset=val,
                      data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer),
                      tokenizer=tokenizer,
                      compute_metrics=get_ner_metrics,
                      callbacks=[EarlyStoppingCallback(early_stopping_patience=3)])
        
    elif args.dataset == WIKIPEDIA_NAME: #MLM
        trainer = WikipediaTrainer(model,
                      training_args,
                      train_dataset=train,
                      eval_dataset=val,
                      data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer),
                      tokenizer=tokenizer,
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
    parser.add_argument('--epochs', type=int, default=2)
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
