#do not need all these imports but keeping them for future 
#copy paste purposes to the training and eval scripts
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

#function to clean text, converting to lower case and removing extra spaces
def clean_text(text): 
