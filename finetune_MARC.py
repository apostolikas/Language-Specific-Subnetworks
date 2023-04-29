from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
#import argparse
import numpy as np
import evaluate

dataset_en = load_dataset("amazon_reviews_multi", "en")
dataset_de = load_dataset("amazon_reviews_multi", "de")
dataset_fr = load_dataset("amazon_reviews_multi", "fr")
dataset_es = load_dataset("amazon_reviews_multi", "es")
dataset_zh = load_dataset("amazon_reviews_multi", "zh")

def process_columns(dataset):
    wanted_columns = ['review_body', 'stars']
    for column_name in dataset['train'].features.keys():
        if column_name not in wanted_columns:
            dataset = dataset.remove_columns(column_name)
    dataset = dataset.rename_column("review_body", "text")
    dataset = dataset.rename_column("stars", "label")
    return dataset

dataset_en = process_columns(dataset_en)
dataset_de = process_columns(dataset_de)
dataset_fr = process_columns(dataset_fr)
dataset_es = process_columns(dataset_es)
dataset_zh = process_columns(dataset_zh)

train_multilingual_dataset = concatenate_datasets([dataset_en['train'], dataset_de['train'], dataset_fr['train'], dataset_es['train'], dataset_zh['train']])
val_multilingual_dataset = concatenate_datasets([dataset_en['validation'], dataset_de['validation'], dataset_fr['validation'], dataset_es['validation'], dataset_zh['validation']])
test_multilingual_dataset = concatenate_datasets([dataset_en['test'], dataset_de['test'], dataset_fr['test'], dataset_es['test'], dataset_zh['test']])


tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
def tokenize_fn(dataset):
    return tokenizer(dataset['text'], padding='max_length', truncation=True)

train_set = train_multilingual_dataset.map(tokenize_fn, batched=True)
train_set = train_set.map(lambda example: {"label": example["label"] - 1})
val_set = val_multilingual_dataset.map(tokenize_fn, batched=True)
val_set = val_set.map(lambda example: {"label": example["label"] - 1})
test_set = test_multilingual_dataset.map(tokenize_fn, batched=True)
test_set = test_set.map(lambda example: {"label": example["label"] - 1})


model = AutoModelForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=5)
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(output_dir="test_trainer",
                                  evaluation_strategy="epoch",
                                  learning_rate=5e-5,
                                  num_train_epochs=1,
                                  logging_strategy="epoch",
                                  seed=42
                                  )

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_set,
    eval_dataset=val_set,
    compute_metrics=compute_metrics,
)

trainer.train()
