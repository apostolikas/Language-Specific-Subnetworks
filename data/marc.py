from transformers import AutoTokenizer
import torch

from .general import ClassificationDataset


class MarcDataset(ClassificationDataset):

    name = 'marc'
    n_classes = 2

    def __init__(self,
                 tokenizer: AutoTokenizer,
                 lang: str = None,
                 split: str = 'train',
                 sample_n: int = 0):
        super().__init__("amazon_reviews_multi", tokenizer, lang, split, sample_n)

    def _get_input(self, row: dict):
        return row['review_body'], row['review_title'], row['product_category']

    def _get_target(self, row: dict):
        targets = torch.tensor(row['stars'] - 1, dtype=torch.long)
        targets[targets == 1] = 0
        targets[targets == 2] = 0
        targets[targets == 3] = 1
        targets[targets == 4] = 1
        return targets
