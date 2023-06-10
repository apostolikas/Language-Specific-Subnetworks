from .general import ALLOWED_LANGUAGES, ClassificationDataset
from .marc import MarcDataset
from .pawsx import PawsXDataset
from .xnli import XNLIDataset
from .wikiann import WikiannDataset
from .wikipedia import WikipediaDataset

WIKIANN_NAME = 'wikiann'
WIKIPEDIA_NAME = 'wikipedia'
ALLOWED_DATASETS = ['marc', 'paws-x', 'xnli',WIKIANN_NAME,WIKIPEDIA_NAME]


def get_dataset(dataset_name: str, *args, **kwargs) -> ClassificationDataset:

    name = dataset_name.lower()

    if name == 'marc':
        return MarcDataset(*args, **kwargs)
    elif name == 'paws-x':
        return PawsXDataset(*args, **kwargs)
    elif name == 'xnli':
        return XNLIDataset(*args, **kwargs)
    elif name == WIKIANN_NAME:
        return WikiannDataset(*args, **kwargs)
    elif name == WIKIPEDIA_NAME:
        return WikipediaDataset(*args, **kwargs)
    else:
        raise NameError(f"{dataset_name} is unknown.")