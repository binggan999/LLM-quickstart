import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from datasets import load_dataset
import datasets

train_data_path = 'datas/YelpReviewFull'    # 训练数据路径
dataset = load_dataset(train_data_path)

dataset

dataset["train"][111]

model_name_or_path = 'models/bert-base-cased'

print(dataset["train"][111])

import random
import pandas as pd
import datasets
from IPython.display import display, HTML
def show_random_elements(dataset, num_examples=10):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)
    
    df = pd.DataFrame(dataset[picks])
    for column, typ in dataset.features.items():
        if isinstance(typ, datasets.ClassLabel):
            df[column] = df[column].transform(lambda i: typ.names[i])
    display(HTML(df.to_html()))
show_random_elements(dataset["train"])

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_datasets = dataset.map(tokenize_function, batched=True)
show_random_elements(tokenized_datasets["train"], num_examples=1)

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42)#.select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42)#.select(range(1000))

from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, num_labels=5) 
#"bert-base-cased"

from transformers import TrainingArguments

model_dir = "models/bert-base-cased-finetune-yelp"

# logging_steps 默认值为500，根据我们的训练数据和步长，将其设置为100
training_args = TrainingArguments(output_dir=model_dir,
                                  per_device_train_batch_size=80,
                                  num_train_epochs=5,
                                  logging_steps=500)
# 完整的超参数配置
print(training_args)

import numpy as np
import evaluate

metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(output_dir=model_dir,
                                  evaluation_strategy="epoch", 
                                  per_device_train_batch_size=64,
                                  num_train_epochs=3,
                                  logging_steps=100)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)
trainer.train()
