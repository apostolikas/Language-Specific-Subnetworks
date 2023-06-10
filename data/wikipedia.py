import torch
from transformers import AutoTokenizer
from datasets import load_dataset, concatenate_datasets, Dataset
from .general import ClassificationDataset


class WikipediaDataset(ClassificationDataset):

    name = 'wikipedia'
    n_classes = 3 #! this is not useful at all here!

    def __init__(self,
                 tokenizer: AutoTokenizer,
                 lang: str = None,
                 split: str = 'train',
                 sample_n: int = 0,
                 **kwargs):
        super().__init__("wikipedia", tokenizer, 'fr', split, sample_n, **kwargs)#todo change to fr

        encodings = self.tokenizer(self.dataset['text'])

        result2 = {}
        for k, t in encodings.items():
            for i in range(0, len(encodings['input_ids']), 128):
                result2[k] = t[i : i + 128]
           

        # result = {
        #     k: t[i : i + 128] for i in range(0, len(encodings['input_ids']), 128)
        #     for k, t in encodings.items()
        # }
        # {i: t[i:i+128] if i+128 < len(encodings['input_ids']) else t[i:len(encodings['input_ids'])-1]
       

    def preprocess_function(self, examples):
        return self.tokenizer([" ".join(x) for x in examples["text"]])
    
    block_size = 128


    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        block_size = 128
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of block_size.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    def _get_input(self, row: dict):
        return row['text']

    def __getitem__(self, i: int):

        row = self.dataset[i]
        inputs = self._get_input(row)

        encodings = self.tokenizer(inputs, truncation=True)
        result = {
            k: [t[i : i + 128] for i in range(0, len(encodings['input_ids']), 128)]
            for k, t in encodings.items()
        }

        result["labels"] = result["input_ids"].copy()
        return result
