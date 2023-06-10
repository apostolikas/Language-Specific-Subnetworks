import random
import os
import argparse
import json
import pickle
ALLOWED_LANGUAGES = ['en', 'de', 'fr', 'es', 'zh']

def get_data_split(start, end, dataset, indices, sample_n):
        split_1_indices = indices[start:end]
        split_list = [dataset[i] for i in split_1_indices] #
        # assert(split_list[2] == dataset[split_1_indices[2]])
        if sample_n != 0:
            split_list = split_list[:sample_n]
        return split_list

def split_data(dataset, sample_n):
    # 1. shuffle 2. split 3. take samples from split

    dataset_length = len(dataset)
    indices = list(range(dataset_length))
    random.seed(0)
    random.shuffle(indices)
    # Divide the indices into three sets
    train_size = int(dataset_length *0.8)
    val_size = int(dataset_length *0.1)

    all_splits = []
    for split in ['train','validation','test']:
        if split == 'train':
            start, end = 0, train_size
        elif split == 'validation':
            start, end = train_size, train_size+val_size
        elif split == 'test':
            start, end = train_size+val_size, len(indices)

        split_list = get_data_split(start, end, dataset, indices, sample_n)
        all_splits.append(split_list)

    return all_splits

def load_dataset(sample_n, langs):
    
    train_all_languages = {}
    val_all_languages = {}
    test_all_languages = {}

    for language in langs:
        path = os.path.join(f'{language}.data.json', 'AA', f"wiki_00")

        train_all_languages[language] = []
        val_all_languages[language] = []
        test_all_languages[language] = []
        with open(path, 'r') as input_file:
            examples = []

            for line in input_file:
                doc = json.loads(line)
                text = doc["text"].strip()
                if text == "":
                    continue
                text = text.replace("\n", " ")
                text = text.replace("[...]", "")
                if "src=" in text: 
                    continue
                examples.append(text) 

            sample_lang = int(sample_n/len(langs))
            train, val, test = split_data(examples, sample_lang)

            train_all_languages[language] = train
            val_all_languages[language] = val
            test_all_languages[language] = test

    return train_all_languages, val_all_languages, test_all_languages

def create_pickle(filename, data):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Simple settings.')
    parser.add_argument('--sample-n',
                        type=int,
                        default=0,
                        help='Number of train samples. 0 means all.')
    args = parser.parse_args()
    langs = ALLOWED_LANGUAGES
    train_all_languages, val_all_languages, test_all_languages = load_dataset(args.sample_n, langs)
    
    create_pickle(f'train_wikipedia.pickle',train_all_languages)
    create_pickle(f'validation_wikipedia.pickle', val_all_languages)
    create_pickle(f'test_wikipedia.pickle', test_all_languages)
