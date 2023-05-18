from datasets import load_dataset, concatenate_datasets, Dataset
import numpy as np
import torch
from transformers import AutoTokenizer
from abc import ABC

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
                 sample_n: int = 0):
        self.dataset_name = dataset_name
        self.tokenizer = tokinizer
        self.langs = [lang] if lang is not None else ALLOWED_LANGUAGES
        self.split = split
        self.dataset = self._load_dataset(sample_n)

    def _load_dataset(self, sample_n: int):
        dataset = concatenate_datasets(
            [load_dataset(self.dataset_name, lang, split=self.split) for lang in self.langs])
        return self._sample(dataset, sample_n)

    def _sample(self, dataset: Dataset, n: int):
        if n == 0 or len(dataset) <= n:
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
        encodings = self.tokenizer(*inputs, padding=True, truncation=True)

        targets = self._get_target(row)

        encodings.update({'labels': targets})
        return encodings
