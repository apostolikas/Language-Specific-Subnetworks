from datasets import load_dataset, concatenate_datasets, Dataset
import numpy as np
import torch
from transformers import AutoTokenizer
from abc import ABC
import random
ALLOWED_LANGUAGES = ['en', 'de', 'fr', 'es', 'zh']


class ClassificationDataset(ABC):

    # implement these in the inherited class!
    name: str = None
    n_classes: int = None

    def __init__(self,
                 dataset_name: str,
                 tokinizer: AutoTokenizer,
                 lang: str = None,
                 split: str = 'train',
                 sample_n: int = 0,
                 tokenizer_kwargs: dict = {},
                 no_load:bool=False):
        self.dataset_name = dataset_name
        self.tokenizer = tokinizer
        self.langs = [lang] if lang is not None else ALLOWED_LANGUAGES
        if dataset_name=='wikipedia':
            self.langs = [f'20220301.{language}' for language in self.langs]
        self.split = split
        if not no_load:
            self.dataset = self._load_dataset(sample_n)
        self.tokenizer_kwargs = tokenizer_kwargs

    def _load_dataset(self, sample_n: int):
        if self.dataset_name=='wikipedia':
            #en, fr, 
            datasets_list = [load_dataset(self.dataset_name, lang, split='train') for lang in self.langs]

            dataset = concatenate_datasets(datasets_list)
            return self.split_data(dataset, sample_n)
        else:
            dataset = concatenate_datasets(
            [load_dataset(self.dataset_name, lang, split=self.split) for lang in self.langs])
            return self._sample(dataset, sample_n)

    def get_data_split(self, start, end, dataset, indices, sample_n):
        split_1_indices = indices[start:end]
        split_list = torch.utils.data.Subset(dataset, split_1_indices)#[dataset[i] for i in split_1_indices]
        assert(split_list[2] == dataset[split_1_indices[2]])
        if sample_n != 0:
            split_list = split_list[:sample_n]
        return split_list

    def split_data(self, dataset, sample_n):
        dataset_length = len(dataset)
        indices = list(range(dataset_length))
        random.shuffle(indices)

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

    def _sample(self, dataset: Dataset, n: int):
        if n == 0:
            return dataset
        choose_n = np.random.choice(len(dataset), n, replace=False).tolist()
        return torch.utils.data.Subset(dataset, choose_n)

    def _get_input(self, row: dict):
        raise NotImplementedError

    def _get_target(self, row: dict):
        raise NotImplementedError

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i: int):

        row = self.dataset[i]
        inputs = self._get_input(row)

        # Tokenize & Encode
        encodings = self.tokenizer(*inputs, padding=True, truncation=True, **self.tokenizer_kwargs)

        targets = self._get_target(row)

        encodings.update({'labels': targets})
        return encodings
