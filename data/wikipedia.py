import torch
from transformers import AutoTokenizer
from datasets import load_dataset, concatenate_datasets, Dataset
import os
import json
import random
import pickle
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
        self.split = split
        with open(f'{split}_wikipedia.pickle', 'rb') as file:
            dict_dataset = pickle.load(file)
        
        dataset = []
        for lang in self.langs:
            dataset.extend(dict_dataset[lang])
        self.dataset = dataset
    
    def __getitem__(self, i: int):

        inputs = self.dataset[i]
        
        encodings = self.tokenizer(inputs, truncation=True)

        return encodings

    def __len__(self):
        return len(self.dataset)