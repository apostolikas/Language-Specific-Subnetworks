import torch
from transformers import AutoTokenizer
from datasets import load_dataset, concatenate_datasets, Dataset
import os
import json
import random
ALLOWED_LANGUAGES = ['en', 'de', 'fr', 'es', 'zh']

class WikipediaDataset():

    name: str = 'wikipedia'
    n_classes: int = None

    def __init__(self,
                 tokenizer: AutoTokenizer,
                 dataset_name: str = 'wikipedia',
                 lang: str = None,
                 split: str = 'train',
                 sample_n: int = 0,
                 tokenizer_kwargs: dict = {},
                 no_load:bool=False
                 ):
        self.dataset_name = dataset_name
        self.tokenizer_kwargs = tokenizer_kwargs
        self.tokenizer = tokenizer
        self.langs = [lang] if lang is not None else ALLOWED_LANGUAGES
        self.langs = ['zh']
        self.split = split
        self.dataset = self.load_dataset(sample_n)

        # super().__init__("wikipedia", tokenizer, 'zh', split, sample_n, **kwargs)#todo change to fr
    def get_data_split(self, start, end, dataset, indices, sample_n):
        split_1_indices = indices[start:end]
        split_list = [dataset[i] for i in split_1_indices] #
        assert(split_list[2] == dataset[split_1_indices[2]])
        if sample_n != 0:
            split_list = split_list[:sample_n]
        return split_list

    def split_data(self, dataset, sample_n):
        # 1. shuffle 2. split 3. take samples from split

        dataset_length = len(dataset)
        indices = list(range(dataset_length))
        random.shuffle(indices)
        self.indices = indices
        # Divide the indices into three sets
        train_size = int(dataset_length *0.8)
        val_size = int(dataset_length *0.1)

        if self.split == 'train':
            start, end = 0, train_size
        elif self.split == 'validation':
            start, end = train_size, train_size+val_size
        elif self.split == 'test':
            start, end = train_size+val_size, len(indices)

        split_list = self.get_data_split(start, end, dataset, indices, sample_n)

        return split_list

    def load_dataset(self, sample_n):
        all_languages_dataset = []
        for i, language in enumerate(self.langs):
            path = os.path.join(f'{language}.data.json', 'AA', f"wiki_00")

            with open(path, 'r') as input_file:
                examples = []

                for line in input_file:
                    doc = json.loads(line)
                    text = doc["text"].strip()
                    if text == "":
                        continue
                    text = text.replace("\n", " ")
                    text = text.replace("[...]", "")
                    if "src=" in text: 
                        continue
                    examples.append(text) 

                sample_lang = int(sample_n/len(self.langs))
                current_split_examples = self.split_data(examples, sample_lang)

                all_languages_dataset.extend(current_split_examples)

        return all_languages_dataset

    def __getitem__(self, i: int):

        inputs = self.dataset[i]
        
        encodings = self.tokenizer(inputs, truncation=True)

        return encodings

    def __len__(self):
        return len(self.dataset)