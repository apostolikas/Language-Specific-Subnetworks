from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

def load_lang_dataset(language, id):
    dataset = load_dataset("xnli", language)

    sets = ['train','validation','test']
    for set_name in sets:
        dataset[set_name] = dataset[set_name].add_column("language_id", [id] * len(dataset[set_name]))
    return dataset

def tokenize_batch(examples):
    dict =  tokenizer(examples['premise'], examples['hypothesis'], padding= False, truncation= True,
                      max_length = 256)
    return dict

'''
 https://colab.research.google.com/github/huggingface/notebooks/blob/master/course/videos/dynamic_padding.ipynb#scrollTo=i1cJ5_qS8MT_
 this way we don't add bos at the beginning of the 2nd sentence but maybe is not needed
 as we have sep id before?
'''
def createDataloader(dataset):
    tokenized_datasets = dataset.map(tokenize_batch, batched = True)
    tokenized_datasets = tokenized_datasets.remove_columns(["premise", "hypothesis"])

    data_collator = DataCollatorWithPadding(tokenizer)
    train_dataloader = DataLoader(
        tokenized_datasets, batch_size=3, shuffle=True, collate_fn=data_collator
    )
    return train_dataloader

def set_seed(seed):
    """
    Function for setting the seed for reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

languages = ["en","de","fr","es","zh"]
languages_datasets = {}
for language_id, language in enumerate(languages):
    languages_datasets[language] = load_lang_dataset(language, language_id)

train_multilingual_dataset = concatenate_datasets([languages_datasets[lang]['train'] for lang in languages_datasets])
val_multilingual_dataset = concatenate_datasets([languages_datasets[lang]['validation'] for lang in languages_datasets])
test_multilingual_dataset = concatenate_datasets([languages_datasets[lang]['test'] for lang in languages_datasets])

train_dataloader = createDataloader(train_multilingual_dataset)
val_dataloader = createDataloader(val_multilingual_dataset)
test_dataloader = createDataloader(test_multilingual_dataset)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = AutoModelForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels = 3)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=2e-05)
set_seed(42)

epochs = 10
for epoch in tqdm(range(epochs)):
    correct_predictions = 0

    losses = []
    model.train()
    for d in train_dataloader:

        labels, language_id, input_ids, attention_mask = d['labels'], d['language_id'], d['input_ids'], d['attention_mask']
        labels, input_ids, attention_mask = labels.to(device), input_ids.to(device), attention_mask.to(device)
        out = model(input_ids = input_ids, attention_mask = attention_mask, labels = labels) 
        loss = out['loss']
        _, preds = torch.max(out['logits'], dim = -1)
        correct_predictions += torch.sum(labels == preds)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        

    correct_predictions_per_lang = {i: 0 for i in range(len(languages))}
    val_correct_predictions = 0
    model.eval()
    for d in val_dataloader:

        labels, language_id, input_ids, attention_mask = d['labels'], d['language_id'], d['input_ids'], d['attention_mask']
        labels, input_ids, attention_mask = labels.to(device), input_ids.to(device), attention_mask.to(device)
        out = model(input_ids = input_ids, attention_mask = attention_mask, labels = labels) 
        loss = out['loss']
        _, preds = torch.max(out['logits'], dim = -1)
        val_correct_predictions += torch.sum(labels == preds)

        for num in range(len(languages)):
            current_lang_pos = torch.where(language_id == num)[0]
            correct_predictions_per_lang[num] += torch.sum(preds[current_lang_pos] == labels[current_lang_pos])
