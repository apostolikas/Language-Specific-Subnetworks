from transformers import AutoTokenizer
from datasets import load_dataset, concatenate_datasets
import torch
from dataclasses import dataclass, field
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollator
import evaluate
import numpy as np

@dataclass
class SmartCollator():
  pad_token_id: int

  def __call__(self, batch):
    batch_inputs = list()
    batch_attention_masks = list()
    labels = list()
    max_size = max([len(ex['input_ids']) for ex in batch])
    for item in batch:
        batch_inputs += [pad_seq(item['input_ids'], max_size, self.pad_token_id)]
        batch_attention_masks += [pad_seq(item['attention_mask'], max_size, 0)]
        labels.append(item['labels'])
    return {"input_ids": torch.tensor(batch_inputs, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
            } 


def pad_seq(seq,max_batch_len, pad_value):
  return seq + (max_batch_len - len(seq))*[pad_value]  


def fetch_datasets():
    dataset_en = load_dataset("paws-x", "en")
    dataset_de = load_dataset("paws-x", "de")
    dataset_fr = load_dataset("paws-x", "fr")
    dataset_es = load_dataset("paws-x", "es")
    dataset_zh = load_dataset("paws-x", "zh")
    train_dataset = concatenate_datasets([dataset_en['train'], dataset_de['train'], dataset_fr['train'], dataset_es['train'], dataset_zh['train']])
    val_dataset = concatenate_datasets([dataset_en['validation'], dataset_de['validation'], dataset_fr['validation'], dataset_es['validation'], dataset_zh['validation']])
    test_dataset = concatenate_datasets([dataset_en['test'], dataset_de['test'], dataset_fr['test'], dataset_es['test'], dataset_zh['test']])
    return train_dataset, val_dataset, test_dataset


tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base', use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained('xlm-roberta-base', num_labels=2)

if __name__ == "__main__":    
    
    train_dataset, val_dataset, test_dataset = fetch_datasets()
    print("--- Data loaded successfully ---")

    def tokenize_data(ex_dataset):
        # Concatenate sentence1 with sentence2 for performance boost 
        text = ex_dataset['sentence1'] + " " + ex_dataset['sentence2']
        # Tokenize & Encode
        encodings = tokenizer.encode_plus(text, 
                                        pad_to_max_length=False,
                                        max_length=512,
                                        add_special_tokens=True,
                                        return_token_type_ids=False,
                                        return_attention_mask=True,
                                        return_overflowing_tokens=False,
                                        return_special_tokens_mask=False,
                                        )
        targets = torch.tensor(ex_dataset['label'],dtype=torch.long)
        encodings.update({'labels': targets})
        return encodings


    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return metric.compute(predictions=predictions, references=labels)


    encoded_train_dataset = list(map(tokenize_data,train_dataset)) #encoding the dataset by mapping the tokenize function to each sample in the dataset
    encoded_val_dataset = list(map(tokenize_data,val_dataset))
    encoded_test_dataset = list(map(tokenize_data,test_dataset))
    print("--- Data processed successfully ---")

    batch_size = 32
    metric_name = "accuracy"

    args = TrainingArguments(  #Defining Training arguments
        output_dir = "Finetune_PAWS-X_results", #Storing model checkpoints 
        seed = 42, 
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=10,
        weight_decay=0.01,
        load_best_model_at_end=True, #During evaluation the model with highest accuracy will be loaded.
        metric_for_best_model=metric_name,
        fp16 = True
    )
    metric = evaluate.load('accuracy')

    trainer = Trainer(
        model,
        args,
        train_dataset= encoded_train_dataset,
        eval_dataset=encoded_val_dataset,
        data_collator=SmartCollator(pad_token_id=tokenizer.pad_token_id),
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
        )       

    "--- Trainer finetune ---"
    trainer.train() #training
    "--- Trainer check model ---"
    trainer.evaluate() #Checking to see if model with highest accuracy is returned
    "--- Trainer evaluate on testset ---"
    results = trainer.evaluate(encoded_test_dataset) #evaluating performance on test dataset. 
    print("Test results : \n" ,results) 


