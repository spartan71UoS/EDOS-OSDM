#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 00:26:56 2023

@author: swetapatra
"""
#pip install sentencepiece
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-v3-base")

data=pd.read_csv("/Users/swetapatra/Downloads/Team_Project_Betterment/augmented_edos_data.csv")
texts=[]
df = pd.DataFrame(data)

# Extract the text sentence from the DataFrame column
text_sentence = df['text'].iloc[0]

# Print the extracted text sentence
#print(text_sentence)

for i in range(len(df)):
    ##print(df['text'].iloc[i])
    ##df['text'].iloc[i]= eda( df['text'].iloc[i], alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1, num_aug=9)
    texts.append(df['text'].iloc[i])


inputs = tokenizer(texts, truncation=True, padding=True, return_tensors="pt")
# Tokenize input texts
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_labels = torch.argmax(logits, dim=1)

# Prepare input tensors
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

# Make predictions
outputs = model(input_ids, attention_mask=attention_mask)
predicted_labels = outputs.logits.argmax(dim=1)

from sklearn.metrics import classification_report, accuracy_score

# Convert tensors to numpy arrays
predicted_labels = predicted_labels.numpy()
print(predicted_labels)
#ground_truth_labels = ground_truth_labels.numpy()

# Calculate accuracy
#accuracy = accuracy_score(ground_truth_labels, predicted_labels)

# Calculate precision, recall, and F1 score
#classification_report = classification_report(ground_truth_labels, predicted_labels)
