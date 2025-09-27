#do not need all these imports but keeping them for future 
#copy paste purposes to the training and eval scripts
import json
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.utils import shuffle
import seaborn as sns
from tkinter import *
from tkinter import messagebox
import sys 
import urllib
import urllib.request

dataset = pd.read_csv("ml-model/datasets/raw/dataset.csv") 

print (dataset.head())

#shuffling the dataset

dataset = shuffle(dataset, random_state=42) 

#testing shuffling
print(dataset.head())

#removing the hyphen from the strings in the csv

for col in dataset.columns:
    dataset[col] = dataset[col].str.replace('_', ' ')

#printing to see if the hyphens are removed
print(dataset.head())

#dataset characteristics 
print(dataset.describe())

#checking the dataset for null values
#null_checker = dataset.apply(lambda x: sum(x.isnull())).to_frame(name='count')
#print(null_checker)

# #plt.figure(figsize=(10,5))
# plt.plot(null_checker.index, null_checker['count']) 
# plt.xticks(null_checker.index, null_checker.index, rotation=45,
#            horizontalalignment='right')
# plt.title("Before removing Null values") 
# plt.xlabel("Column names") 
# plt.margins(0.1) 
# plt.show()  

#severityDataset = pd.read_csv("ml-model/datasets/raw/severity.csv") 

#print (severityDataset.head())

#cleaning and preprocessing the dataset for the huggingface trainer 

#identifying all symptom columns
symptom_columns = [col for col in dataset.columns if "Symptom" in col]

#rmocing extra spaces and converting to lowercase
def clean_text(text): 
    text = str(text).lower() 
    text = re.sub(r"\s+", " ", text)   # replace multiple spaces with a single space
    return text.strip() 

#combining symptoms into one text string 
dataset['text'] = dataset[symptom_columns].apply ( 
    lambda row: ' '.join([clean_text(str(item)) for item in row if pd.notnull(item)]), axis=1
)

#clean disease names so that they match the same formattting as symptoms
dataset["label"] = dataset["Disease"].apply(clean_text) 

#encode each disease name into integers since that is what clincalBERT expects
label_encoder = LabelEncoder() 
dataset['label'] = label_encoder.fit_transform(dataset['label'])

#previewing the cleaned dataset before saving into the processed folder
print ("----------- PROCESSED DATASET 10 EXAMPLES-----------") 
#format should be | row number | list of symptoms | disease encoded as an integer 
print (dataset[['text', 'label']].head(10))

#need to find out what the mapping is between the encoded diseases and the integers 

label_mapping = {disease: int(label) for disease, label in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))}

#save the mapping to a json file for future reference
with open('ml-model/datasets/processed/label_mapping.json', 'w') as f:
    json.dump(label_mapping, f, indent = 4) 

#print the first few mappins to ensure correctness
print ("----------- LABEL MAPPING EXAMPLES -----------")
for disease, label in list(label_mapping.items())[:5]:
    print(f" | {disease} | {label} |")

# ---------------- SAVE CLEANED DATASET ----------------
dataset[['text', 'label']].to_csv("ml-model/datasets/processed/dataset.csv", index=False)
print("\n✅ Preprocessed dataset saved to ml-model/datasets/processed/dataset.csv")
print("✅ Label mapping saved to ml-model/datasets/processed/label_mapping.json")