from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, PreTrainedTokenizer
#import argparse
import torch
from dataclasses import dataclass
from torch.utils.data.dataset import Dataset
from typing import Dict
from typing import List
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
# def tokenize_fn(dataset):
#     return tokenizer(dataset['text'], padding='max_length', truncation=True)
# train_set = train_multilingual_dataset.map(tokenize_fn, batched=True)
# train_set = train_set.map(lambda example: {"label": example["label"] - 1})
# val_set = val_multilingual_dataset.map(tokenize_fn, batched=True)
# val_set = val_set.map(lambda example: {"label": example["label"] - 1})
# test_set = test_multilingual_dataset.map(tokenize_fn, batched=True)
# test_set = test_set.map(lambda example: {"label": example["label"] - 1})

train_set = train_multilingual_dataset.map(lambda example: {"label": example["label"] - 1})
val_set = val_multilingual_dataset.map(lambda example: {"label": example["label"] - 1})
test_set = test_multilingual_dataset.map(lambda example: {"label": example["label"] - 1})


@dataclass
class Example:
    text: str
    label: int

#train_set = [Example(text=item["premise"], text_b=item["hypothesis"], label=item["label"]) for item in train_set["train"]]

train_dataset = [Example(text=item["text"], label=item["label"]) for item in train_set]
val_dataset = [Example(text=item["text"], label=item["label"]) for item in val_set]

@dataclass
class Features:
    input_ids: List[int]
    attention_mask: List[int]
    label: int

class TextDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, 
                 pad_to_max_length: bool, 
                 max_len: int,
                 examples: List[Example]) -> None:
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.examples: List[Example] = examples
        self.current = 0
        self.pad_to_max_length = pad_to_max_length

    def encode(self, ex: Example) -> Features:
        encode_dict = self.tokenizer.encode_plus(text=ex.text,
                                                 add_special_tokens=True,
                                                 max_length=self.max_len,
                                                 pad_to_max_length=self.pad_to_max_length,
                                                 return_token_type_ids=False,
                                                 return_attention_mask=True,
                                                 return_overflowing_tokens=False,
                                                 return_special_tokens_mask=False,
                                                 )
        return Features(input_ids=encode_dict["input_ids"],
                        attention_mask=encode_dict["attention_mask"],
                        label=ex.label)

    def __getitem__(self, idx) -> Features:
        return self.encode(ex=self.examples[idx])

    def __len__(self):
        return len(self.examples)


def pad_seq(seq: List[int], max_batch_len: int, pad_value: int) -> List[int]:
    return seq + (max_batch_len - len(seq)) * [pad_value]

@dataclass
class CustomTransformation():
    pad_token_id: int

    def __call__(self, batch: List[Features]) -> Dict[str, torch.Tensor]:
        batch_inputs = list()
        batch_attention_masks = list()
        labels = list()
        max_size = max([len(ex.input_ids) for ex in batch])
        for item in batch:
            batch_inputs += [pad_seq(item.input_ids, max_size, self.pad_token_id)]
            batch_attention_masks += [pad_seq(item.attention_mask, max_size, 0)]
            labels.append(item.label)

        return {"input_ids": torch.tensor(batch_inputs, dtype=torch.long),
                "attention_mask": torch.tensor(batch_attention_masks, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long)
                }

model = AutoModelForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=5)
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(output_dir="test_trainer",
                                  evaluation_strategy='steps',
                                  eval_steps=2000,
                                  learning_rate=5e-5,
                                  num_train_epochs=1,
                                  logging_steps=1000,
                                  seed=42
                                  )

train_data = TextDataset(tokenizer=tokenizer,
                        max_len=512,
                        examples=train_dataset,
                        pad_to_max_length=False)  
                        
valid_data = TextDataset(tokenizer=tokenizer,
                        max_len=512,
                        examples=val_dataset,
                        pad_to_max_length=False)  


# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_set,
#     eval_dataset=val_set,
#     compute_metrics=compute_metrics,
# )

train_collator = CustomTransformation(pad_token_id=tokenizer.pad_token_id)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    data_collator=train_collator,
    eval_dataset=valid_data,
    compute_metrics=compute_metrics,
)

trainer.train()
