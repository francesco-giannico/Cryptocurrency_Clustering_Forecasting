import os
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from utility.folder_creator import folder_creator

PATH_CLEANED_FOLDER="../preparation/preprocessed_dataset/cleaned/final/"
PATH_NORMALIZED_FOLDER="../preparation/preprocessed_dataset/constructed/normalized/"

#SCALING
"""Normalization is the process of scaling individual samples to have unit norm. 
This process can be useful if you plan to use a quadratic form such 
as the dot-product or any other kernel to quantify the similarity of any pair of samples.
This assumption is the base of the Vector Space Model often used in text classification and clustering contexts.
If you want to cluster based on similar shape in the cluster rather then similar variance (standardization)"""
def normalize():
    folder_creator("../preparation/preprocessed_dataset/constructed/",1)
    folder_creator(PATH_NORMALIZED_FOLDER, 1)
    excluded_features = ['Date']
    for crypto in os.listdir(PATH_CLEANED_FOLDER):
        df = pd.read_csv(PATH_CLEANED_FOLDER+crypto,delimiter=',', header=0)
        scaler = MinMaxScaler()
        for col in df.columns:
            if col not in excluded_features:
                normalized = scaler.fit_transform(df[col].values.reshape(-1, 1))
                df[col] = pd.Series(normalized.reshape(-1))
        df.to_csv( PATH_NORMALIZED_FOLDER+crypto,sep=",", index=False)