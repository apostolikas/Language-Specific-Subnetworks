from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset
from evaluate import evaluator
import evaluate
import pandas as pd
import torch

model = AutoModelForSequenceClassification.from_pretrained("xlm-roberta-base")
tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base', use_fast=True)

dataset_en = load_dataset("amazon_reviews_multi", "en")
dataset_en = dataset_en['test']

dataset_de = load_dataset("amazon_reviews_multi", "de")
dataset_de = dataset_de['test']

dataset_fr = load_dataset("amazon_reviews_multi", "fr")
dataset_fr = dataset_fr['test']

dataset_es = load_dataset("amazon_reviews_multi", "es")
dataset_es = dataset_es['test']

dataset_zh = load_dataset("amazon_reviews_multi", "zh")
dataset_zh = dataset_zh['test']


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
    #targets = torch.tensor(ex_dataset['stars']-1,dtype=torch.long)
    targets = ex_dataset['stars'] - 1
    encodings.update({'labels': targets})
    return encodings

testset_en = list(map(tokenize_data,dataset_en)) 
testset_de = list(map(tokenize_data,dataset_de)) 
testset_fr = list(map(tokenize_data,dataset_fr)) 
testset_es = list(map(tokenize_data,dataset_es)) 
testset_zh = list(map(tokenize_data,dataset_zh)) 

testset_en = Dataset.from_pandas(pd.DataFrame(data=testset_en))
testset_en = testset_en[['input_ids','attention_mask','labels']]

testset_de = Dataset.from_pandas(pd.DataFrame(data=testset_de))
testset_de = testset_de[['input_ids','attention_mask','labels']]

testset_fr = Dataset.from_pandas(pd.DataFrame(data=testset_fr))
testset_fr = testset_fr[['input_ids','attention_mask','labels']]

testset_es = Dataset.from_pandas(pd.DataFrame(data=testset_es))
testset_es = testset_es[['input_ids','attention_mask','labels']]

testset_zh = Dataset.from_pandas(pd.DataFrame(data=testset_zh))
testset_zh = testset_zh[['input_ids','attention_mask','labels']]


metric = evaluate.load("accuracy")
task_evaluator = evaluator("text-classification")

results_en = task_evaluator.compute(model_or_pipeline=model, data=testset_en, metric=metric)          
print("English : " ,results_en)
results_de = task_evaluator.compute(model_or_pipeline=model, data=testset_de, metric=metric)          
print("German : " ,results_de)
results_fr = task_evaluator.compute(model_or_pipeline=model, data=testset_fr, metric=metric)          
print("French : " ,results_fr)
results_es = task_evaluator.compute(model_or_pipeline=model, data=testset_es, metric=metric)          
print("Spanish : " ,results_es)
results_zh = task_evaluator.compute(model_or_pipeline=model, data=testset_zh, metric=metric)          
print("Chinese : " ,results_zh)




