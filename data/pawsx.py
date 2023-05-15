from transformers import AutoTokenizer
import torch

from .general import ClassificationDataset


class PawsXDataset(ClassificationDataset):

    name = 'paws-x'
    n_classes = 2

    def __init__(self,
                 tokenizer: AutoTokenizer,
                 lang: str = None,
                 split: str = 'train',
                 sample_n: int = 0):
        super().__init__("paws-x", tokenizer, lang, split, sample_n)

    def _get_input(self, row: dict):
        return row['sentence1'], row['sentence2']

    def _get_target(self, row: dict):
        targets = torch.tensor(row['label'], dtype=torch.long)
        return targets
