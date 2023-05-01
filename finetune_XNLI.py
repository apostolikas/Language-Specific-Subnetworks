from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, Trainer, DataCollator
from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from tqdm import tqdm
import evaluate
import math

def load_lang_dataset(language, id):
    dataset = load_dataset("xnli", language)

    #todo useful for accuracy per language
    # sets = ['train','validation','test']
    # for set_name in sets:
    #     dataset[set_name] = dataset[set_name].add_column("language_id", [id] * len(dataset[set_name]))
    return dataset

def tokenize_batch(examples):
    dict =  tokenizer(examples['premise'], examples['hypothesis'], padding= False, truncation= True,
                      max_length = 256)
    return dict

def tokenize_dataset(dataset):
    tokenized_datasets = dataset.map(tokenize_batch, batched = True)
    tokenized_datasets = tokenized_datasets.remove_columns(["premise", "hypothesis"])
    return tokenized_datasets

class DummyDataset(data.Dataset):

    def __init__(self, input_ids, attention_mask, labels):
        """
        Inputs:
            size - Number of data points we want to generate
            std - Standard deviation of the noise (see generate_continuous_xor function)
        """
        super().__init__()
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        # Number of data point we have. Alternatively self.data.shape[0], or self.label.shape[0]
        return len(self.labels)

    def __getitem__(self, idx):
        # Return the idx-th data point of the dataset
        # If we have multiple things to return (data point and label), we can return them as tuple
        input_ids = self.input_ids[idx]
        attention_mask = self.attention_mask[idx]
        label = self.labels[idx]
        # return input_ids, attention_mask, label
        return {"input_ids": input_ids, "labels": label,'attention_mask':attention_mask}

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

languages = ["en","de","fr","es","zh"]
languages_datasets = {}
for language_id, language in enumerate(languages):
    languages_datasets[language] = load_lang_dataset(language, language_id)

train_multilingual_dataset = concatenate_datasets([languages_datasets[lang]['train'] for lang in languages_datasets])
val_multilingual_dataset = concatenate_datasets([languages_datasets[lang]['validation'] for lang in languages_datasets])
test_multilingual_dataset = concatenate_datasets([languages_datasets[lang]['test'] for lang in languages_datasets])

tokenized_train_dataset = tokenize_dataset(train_multilingual_dataset)
tokenized_val_dataset = tokenize_dataset(val_multilingual_dataset)
tokenized_test_dataset = tokenize_dataset(test_multilingual_dataset)
batch_size = 104#128 too big
# input_ids = [[i for i in range(256)] for j in range(batch_size)]
# attention_mask = [[1 for i in range(256)] for j in range(batch_size)]
# labels = [1 for j in range(batch_size)]
# tokenized_train_dataset = DummyDataset(input_ids, attention_mask, labels)
# tokenized_val_dataset = DummyDataset(input_ids, attention_mask, labels)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = AutoModelForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels = 3)
model = model.to(device)

metric_name = "accuracy"
num_batches_per_epoch = math.ceil(len(tokenized_train_dataset) / batch_size)
num_epochs = 10

args = TrainingArguments(  #Defining Training arguments
    output_dir = "Finetune_XNLI-X_results", #Storing model checkpoints 
    seed = 42, 
    evaluation_strategy = "steps",#"epoch", #Evaluation is done (and logged) every eval_steps
    save_strategy = "steps",#"epoch", #Save is done every save_steps
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    max_steps = num_batches_per_epoch*num_epochs, #total number of training steps to perform. 
    eval_steps=math.ceil(num_batches_per_epoch/3),
    save_steps=math.ceil(num_batches_per_epoch/3),#number of updates steps before two checkpoint saves
    load_best_model_at_end=True, #During evaluation the model with highest accuracy will be loaded.
    metric_for_best_model=metric_name,
    save_total_limit =1,#limit the total amount of checkpoints
    fp16 = True#True,
)
metric = evaluate.load('accuracy')

trainer = Trainer(
    model,
    args,
    train_dataset= tokenized_train_dataset,
    eval_dataset= tokenized_val_dataset,
    data_collator=DataCollatorWithPadding(tokenizer,pad_to_multiple_of=8), #multiple of 8 for float16 is better
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)     
print('Device ',args.device)
"--- Trainer finetune ---"
trainer.train() #training
"--- Trainer check model ---"
trainer.evaluate() #Checking to see if model with highest accuracy is returned
"--- Trainer evaluate on testset ---"

results = trainer.evaluate(tokenized_test_dataset) #evaluating performance on test dataset. 
# print("Test results : \n" ,results) 