#importing libraries
import os
import pandas as pd
from  sklearn.model_selection import train_test_split
from torch import tensor
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

print (f"Training samples: {len(train_dataset)}")
print (f"Testing samples: {len(test_dataset)}")
    

#loading the tokenizer and model from huggingface
model_name = "emilyalsentzer/Bio_ClinicalBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels= len(processed_dataset['label'].unique()) #processed_dataset['label'] selects the label column from teh processed dataset and .unique() gets the unique values in that column
    #clinicalBERT requires to know how many labels are in the dataset since that corresponds with how many diseases it can classify given symptoms into
)

#encode text into token IDS
def encode_batch(batch): 
    return tokenizer(batch['text'], padding= "max_length", truncation=True, max_length=128)
#trunction = true, ensures text longer than the max length is cut off to fit the model
#max_length = 128 sets the maximum length of the tokenized input to 128 tokens
#padding = "max_length" pads shorter sequences to the maximum length
#dynamic padding can be better for memory efficiency -- will consider changing to dynamic padding


#convert pandas DataFrames into Hugging Face Dataset objects
train_dataset = Dataset.from_pandas(train_dataset)
test_dataset = Dataset.from_pandas(test_dataset)

#the original dataset would have looked like
# skin rash blackheads scurring      1
# after converting to a Hugging Face Dataset object, it looks like
#{'text': 'skin rash blackheads scurring', 
# 'label': 1
#}

# Apply the tokenizer to each dataset
# batched=True → function gets a batch of rows at once (much faster)
# After this step, new columns 'input_ids' and 'attention_mask' are added
train_dataset = train_dataset.map(encode_batch, batched=True)
test_dataset = test_dataset.map(encode_batch, batched=True)

#{
 # 'text': 'skin rash blackheads scurring',
  #'label': 1,
  #'input_ids': [101, 4678, 6829, 16662, 22324, 102, 0, 0, 0, ...],  # token IDs
  #'attention_mask': [1, 1, 1, 1, 1, 1, 0, 0, 0, ...]              # 1 for real tokens, 0 for padding
#}
#input_ids = integers that map words to ClinicalBERT’s vocabulary
#attention_mask = tells model which tokens are padding

# Hugging Face Trainer expects labels in a column named "labels"
# Rename "label" → "labels" to avoid KeyErrors during training
#setting the format of the dataset to comply with hugging face trainer
train_dataset = train_dataset.rename_column("label", "labels")
test_dataset = test_dataset.rename_column("label", "labels")
                                          

#ensure the dataset is in the right format for pytorch
train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

#for theabove line
# Convert dataset columns into PyTorch tensors
# Now each item will look like:
# {
#   "input_ids": tensor([...]),
#   "attention_mask": tensor([...]),
#   "labels": tensor(label_id)
# }

#this is what a single row shownn earlier looks like at the end of this section of code
#Now each row is a PyTorch tensor dictionary:
#{
  #'input_ids': tensor([101, 4678, 6829, ...]),
  #'attention_mask': tensor([1, 1, 1, 1, ...]),
  #'labels': tensor(1)
#}
