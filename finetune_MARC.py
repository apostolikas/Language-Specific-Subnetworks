from transformers import AutoTokenizer
from datasets import load_dataset, concatenate_datasets, Dataset
import torch
from dataclasses import dataclass
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollator
import evaluate
import numpy as np

def concatenate_dictionaries(dict1, dict2, dict3, dict4, dict5):
    concatenated_dict = {key: dict1[key] + dict2[key] + dict3[key] + dict4[key] + dict5[key] for key in dict1}
    return concatenated_dict

@dataclass
class SmartCollator():
  pad_token_id: int

  def __call__(self, batch):
    batch_inputs = list()
    batch_attention_masks = list()
    labels = list()
    max_size = max([len(ex['input_ids']) for ex in batch])
    for item in batch:
        batch_inputs += [pad_seq(item['input_ids'], max_size, self.pad_token_id)]
        batch_attention_masks += [pad_seq(item['attention_mask'], max_size, 0)]
        labels.append(item['labels'])
    return {"input_ids": torch.tensor(batch_inputs, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
            } 

def pad_seq(seq,max_batch_len, pad_value):
  return seq + (max_batch_len - len(seq))*[pad_value]  


def fetch_datasets():
    dataset_en = load_dataset("amazon_reviews_multi", "en")
    dataset_de = load_dataset("amazon_reviews_multi", "de")
    dataset_fr = load_dataset("amazon_reviews_multi", "fr")
    dataset_es = load_dataset("amazon_reviews_multi", "es")
    dataset_zh = load_dataset("amazon_reviews_multi", "zh")

    # testset_en = dataset_en['test']
    # testset_de = dataset_de['test']
    # testset_fr = dataset_fr['test']
    # testset_es = dataset_es['test']
    # testset_zh = dataset_zh['test']

    # validset_en = dataset_en['validation']
    # validset_de = dataset_de['validation']
    # validset_fr = dataset_fr['validation']
    # validset_es = dataset_es['validation']
    # validset_zh = dataset_zh['validation']

    testset_en = Dataset.from_dict(concatenate_dictionaries(dataset_en['test'][:300],dataset_en['test'][1000:1300],dataset_en['test'][2000:2300],dataset_en['test'][3000:3300],dataset_en['test'][4000:4300])).shuffle(seed=42)
    testset_de = Dataset.from_dict(concatenate_dictionaries(dataset_de['test'][:300],dataset_de['test'][1000:1300],dataset_de['test'][2000:2300],dataset_de['test'][3000:3300],dataset_de['test'][4000:4300])).shuffle(seed=42)
    testset_fr = Dataset.from_dict(concatenate_dictionaries(dataset_fr['test'][:300],dataset_fr['test'][1000:1300],dataset_fr['test'][2000:2300],dataset_fr['test'][3000:3300],dataset_fr['test'][4000:4300])).shuffle(seed=42)
    testset_es = Dataset.from_dict(concatenate_dictionaries(dataset_es['test'][:300],dataset_es['test'][1000:1300],dataset_es['test'][2000:2300],dataset_es['test'][3000:3300],dataset_es['test'][4000:4300])).shuffle(seed=42)
    testset_zh = Dataset.from_dict(concatenate_dictionaries(dataset_zh['test'][:300],dataset_zh['test'][1000:1300],dataset_zh['test'][2000:2300],dataset_zh['test'][3000:3300],dataset_zh['test'][4000:4300])).shuffle(seed=42)

    validset_en = Dataset.from_dict(concatenate_dictionaries(dataset_en['validation'][:300],dataset_en['validation'][1000:1300],dataset_en['validation'][2000:2300],dataset_en['validation'][3000:3300],dataset_en['validation'][4000:4300])).shuffle(seed=42)
    validset_de = Dataset.from_dict(concatenate_dictionaries(dataset_de['validation'][:300],dataset_de['validation'][1000:1300],dataset_de['validation'][2000:2300],dataset_de['validation'][3000:3300],dataset_de['validation'][4000:4300])).shuffle(seed=42)
    validset_fr = Dataset.from_dict(concatenate_dictionaries(dataset_fr['validation'][:300],dataset_fr['validation'][1000:1300],dataset_fr['validation'][2000:2300],dataset_fr['validation'][3000:3300],dataset_fr['validation'][4000:4300])).shuffle(seed=42)
    validset_es = Dataset.from_dict(concatenate_dictionaries(dataset_es['validation'][:300],dataset_es['validation'][1000:1300],dataset_es['validation'][2000:2300],dataset_es['validation'][3000:3300],dataset_es['validation'][4000:4300])).shuffle(seed=42)
    validset_zh = Dataset.from_dict(concatenate_dictionaries(dataset_zh['validation'][:300],dataset_zh['validation'][1000:1300],dataset_zh['validation'][2000:2300],dataset_zh['validation'][3000:3300],dataset_zh['validation'][4000:4300])).shuffle(seed=42)
             
                     
    train_dataset = concatenate_datasets([Dataset.from_dict(concatenate_dictionaries(dataset_en['train'][:8_000],dataset_en['train'][40_000:48_000],dataset_en['train'][80_000:88_000],dataset_en['train'][120_000:128_000],dataset_en['train'][160_000:168_000])), 
                                        Dataset.from_dict(concatenate_dictionaries(dataset_de['train'][:8_000],dataset_de['train'][40_000:48_000],dataset_de['train'][80_000:88_000],dataset_de['train'][120_000:128_000],dataset_de['train'][160_000:168_000])), 
                                        Dataset.from_dict(concatenate_dictionaries(dataset_fr['train'][:8_000],dataset_fr['train'][40_000:48_000],dataset_fr['train'][80_000:88_000],dataset_fr['train'][120_000:128_000],dataset_fr['train'][160_000:168_000])), 
                                        Dataset.from_dict(concatenate_dictionaries(dataset_es['train'][:8_000],dataset_es['train'][40_000:48_000],dataset_es['train'][80_000:88_000],dataset_es['train'][120_000:128_000],dataset_es['train'][160_000:168_000])), 
                                        Dataset.from_dict(concatenate_dictionaries(dataset_zh['train'][:8_000],dataset_zh['train'][40_000:48_000],dataset_zh['train'][80_000:88_000],dataset_zh['train'][120_000:128_000],dataset_zh['train'][160_000:168_000]))])

    val_dataset = concatenate_datasets([validset_en,validset_de,validset_fr,validset_es,validset_zh])
    test_dataset = concatenate_datasets([testset_en,testset_de,testset_fr,testset_es,testset_zh])

    train_dataset = train_dataset.shuffle(seed=42)
    val_dataset = val_dataset
    test_dataset = test_dataset

    #train_dataset = concatenate_datasets([dataset_en['train'], dataset_de['train'], dataset_fr['train'], dataset_es['train'], dataset_zh['train']])
    #val_dataset = concatenate_datasets([dataset_en['validation'], dataset_de['validation'], dataset_fr['validation'], dataset_es['validation'], dataset_zh['validation']])
    #test_dataset = concatenate_datasets([dataset_en['test'], dataset_de['test'], dataset_fr['test'], dataset_es['test'], dataset_zh['test']])

    return train_dataset, val_dataset, test_dataset , validset_en, validset_de,validset_fr, validset_es,validset_zh ,testset_en, testset_de, testset_fr, testset_es, testset_zh


tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base', use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained('xlm-roberta-base', num_labels=5)

if __name__ == "__main__":    
    
    train_dataset, val_dataset, test_dataset, validset_en, validset_de,validset_fr, validset_es,validset_zh ,testset_en, testset_de, testset_fr, testset_es, testset_zh = fetch_datasets()
    print("--- Data loaded successfully ---")

    def tokenize_data(ex_dataset):

        # Concatenate review body with review_title and product_category for performance boost 
        text = ex_dataset['review_body'] + " " + ex_dataset['review_title'] + " " + ex_dataset['product_category']
        # Tokenize & Encode
        encodings = tokenizer.encode_plus(text, 
                                        pad_to_max_length=False,
                                        max_length=512,
                                        add_special_tokens=True,
                                        return_token_type_ids=False,
                                        return_attention_mask=True,
                                        return_overflowing_tokens=False,
                                        return_special_tokens_mask=False,
                                        )
        
        # Labels in range 0-4
        targets = torch.tensor(ex_dataset['stars']-1,dtype=torch.long)
        encodings.update({'labels': targets})
        return encodings


    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)

        return metric.compute(predictions=predictions, references=labels)

    
    encoded_train_dataset = list(map(tokenize_data,train_dataset)) #encoding the dataset by mapping the tokenize function to each sample in the dataset
    encoded_val_dataset = list(map(tokenize_data,val_dataset))
    encoded_test_dataset = list(map(tokenize_data,test_dataset))

    testset_en = list(map(tokenize_data,testset_en))
    testset_de = list(map(tokenize_data,testset_de))
    testset_fr = list(map(tokenize_data,testset_fr))
    testset_es = list(map(tokenize_data,testset_es))
    testset_zh = list(map(tokenize_data,testset_zh))

    validset_en = list(map(tokenize_data,validset_en))
    validset_de = list(map(tokenize_data,validset_de))
    validset_fr = list(map(tokenize_data,validset_fr))
    validset_es = list(map(tokenize_data,validset_es))
    validset_zh = list(map(tokenize_data,validset_zh))

    print("--- Data processed successfully ---")

    batch_size = 32
    metric_name = "accuracy"

    args = TrainingArguments(  #Defining Training arguments
        output_dir = "Finetune_MARC_results", #Storing model checkpoints 
        seed = 42, 
        evaluation_strategy = "epoch",
        save_strategy = 'epoch',
        learning_rate=5e-5, #2e-5
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=5, #5
        weight_decay=0.01,
        load_best_model_at_end=True, #During evaluation the model with highest accuracy will be loaded.
        metric_for_best_model=metric_name,
        #eval_steps = 2000,
        #save_steps = 12000,
        fp16 = True
    )
    metric = evaluate.load('accuracy')

    trainer = Trainer(
        model,
        args,
        train_dataset= encoded_train_dataset,
        eval_dataset=encoded_val_dataset,
        data_collator=SmartCollator(pad_token_id=tokenizer.pad_token_id),
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
        )       

    "--- Trainer finetune ---"
    trainer.train() #training
    "--- Trainer check model ---"
    trainer.evaluate() #Checking to see if model with highest accuracy is returned
    print(trainer.state.best_model_checkpoint)
    "--- Trainer evaluate on testset ---"
    results = trainer.evaluate(encoded_test_dataset) #evaluating performance on test dataset. 
    print("Acc on concatenated test dataset : ",results)

    "--- Separate valid accs ---"
    results_val_en = trainer.evaluate(validset_en) 
    print("Valid Acc - English : ",results_val_en)

    results_val_de = trainer.evaluate(validset_de) 
    print("Valid Acc - German : ",results_val_de)

    results_val_fr = trainer.evaluate(validset_fr) 
    print("Valid Acc - French : ",results_val_fr)

    results_val_es = trainer.evaluate(validset_es) 
    print("Valid Acc - Spanish : ",results_val_es)

    results_val_zh = trainer.evaluate(validset_zh) 
    print("Valid Acc - Chinese : ",results_val_zh)

    "--- Separate test accs ---"
    results_test_en = trainer.evaluate(testset_en) 
    print("Test Acc - English : ",results_test_en)

    results_test_de = trainer.evaluate(testset_de) 
    print("Test Acc - German : ",results_test_de)

    results_test_fr = trainer.evaluate(testset_fr) 
    print("Test Acc - French : ",results_test_fr)

    results_test_es = trainer.evaluate(testset_es) 
    print("Test Acc - Spanish : ",results_test_es)

    results_test_zh = trainer.evaluate(testset_zh) 
    print("Test Acc - Chinese : ",results_test_zh)



