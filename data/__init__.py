from .general import ALLOWED_LANGUAGES, ClassificationDataset
from .marc import MarcDataset
from .pawsx import PawsXDataset
from .xnli import XNLIDataset

ALLOWED_DATASETS = ['marc', 'paws-x', 'xnli']


def get_dataset(dataset_name: str, *args, **kwargs) -> ClassificationDataset:

    name = dataset_name.lower()

    if name == 'marc':
        return MarcDataset(*args, **kwargs)
    elif name == 'paws-x':
        return PawsXDataset(*args, **kwargs)
    elif name == 'xnli':
        return XNLIDataset(*args, **kwargs)
    else:
        raise NameError(f"{dataset_name} is unknown.")