#importing libraries
import os
import pandas as pd
from  sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset 

#loading the preprocessed dataset
processed_dataset_path = "ml-model/datasets/processed/processed_dataset.csv"
processed_dataset = pd.read_csv(processed_dataset_path)

#we need to split the dataset into training and a test split 
#error catching if the file does not exist 
if not os.path.exists(processed_dataset_path):
    raise FileNotFoundError(f"Processed dataset not found at {processed_dataset_path}. Please run preprocess.py first.")

train_dataset, test_dataset = train_test_split(
    processed_dataset, 
    test_size=0.2,
    random_state=42,
    stratify=processed_dataset['label']
) 

print (f"Training samples: {len(train_dataset)})")
print (f"Testing samples: {len(test_dataset)})")
       
