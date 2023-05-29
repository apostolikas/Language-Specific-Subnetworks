import torch
from transformers import AutoTokenizer

from .general import ClassificationDataset


class XNLIDataset(ClassificationDataset):

    name = 'xnli'
    n_classes = 3

    def __init__(self,
                 tokenizer: AutoTokenizer,
                 lang: str = None,
                 split: str = 'train',
                 sample_n: int = 0,
                 **kwargs):
        super().__init__("xnli", tokenizer, lang, split, sample_n, **kwargs)

    def _get_input(self, row: dict):
        return row['premise'], row['hypothesis']

    def _get_target(self, row: dict):
        targets = torch.tensor(row['label'], dtype=torch.long)
        return targets
