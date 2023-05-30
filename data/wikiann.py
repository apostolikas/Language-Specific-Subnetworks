from transformers import AutoTokenizer
import torch

from .general import ClassificationDataset

class WikiannDataset(ClassificationDataset):

    name = 'wikiann'
    n_classes = 7

    def __init__(self,
                 tokenizer: AutoTokenizer,
                 lang: str = None,
                 split: str = 'train',
                 sample_n: int = 0,**kwargs):
        super().__init__('wikiann', tokenizer, lang, split, sample_n, **kwargs)

    def _get_input(self, row: dict):
        return row['tokens']

    def align_labels_with_tokens(self,labels, word_ids):
        new_labels = []
        current_word = None
        for word_id in word_ids:
            if word_id != current_word:
                # Start of a new word!
                current_word = word_id
                label = -100 if word_id is None else labels[word_id]
                new_labels.append(label)
            elif word_id is None:
                # Special token
                new_labels.append(-100)
            else:
                # Same word as previous token
                label = labels[word_id]
                # If the label is B-XXX we change it to I-XXX
                if label % 2 == 1:
                    label += 1
                new_labels.append(label)
        return new_labels
    
    def tokenize_and_align_labels(self, example):
        tokenized_inputs = self.tokenizer(
            example["tokens"], truncation=True, is_split_into_words=True
        )
        all_labels = example["ner_tags"]
      
        word_ids = tokenized_inputs.word_ids()

        tokenized_inputs["labels"] = self.align_labels_with_tokens(all_labels, word_ids)
        return tokenized_inputs
    
    def _get_target(self, row: dict):
        targets = torch.tensor(row['label'], dtype=torch.long)
        return targets
    
    def __getitem__(self, i: int):
        row = self.dataset[i]
        encodings = self.tokenize_and_align_labels(row)

        return encodings