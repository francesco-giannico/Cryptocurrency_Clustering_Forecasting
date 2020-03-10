import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler
import pandas as pd
from utility.folder_creator import folder_creator


#SCALING
"""Normalization is the process of scaling individual samples to have unit norm. 
This process can be useful if you plan to use a quadratic form such 
as the dot-product or any other kernel to quantify the similarity of any pair of samples.
This assumption is the base of the Vector Space Model often used in text classification and clustering contexts.
If you want to cluster based on similar shape in the cluster rather then similar variance (standardization)"""
def min_max_scaling(input_path,output_path):
    folder_creator(output_path,1)
    excluded_features = ['Date']
    for crypto in os.listdir(input_path):
        df = pd.read_csv(input_path+crypto,delimiter=',', header=0)
        scaler = MinMaxScaler()
        for col in df.columns:
            if col not in excluded_features:
                normalized = scaler.fit_transform(df[col].values.reshape(-1, 1))
                df[col] = pd.Series(normalized.reshape(-1))
        df.to_csv(output_path+crypto,sep=",", index=False)


def max_abs_scaling(input_path,output_path):
    folder_creator(output_path,1)
    excluded_features = ['Date']
    for crypto in os.listdir(input_path):
        df = pd.read_csv(input_path+crypto,delimiter=',', header=0)
        scaler = MaxAbsScaler()
        for col in df.columns:
            if col not in excluded_features:
                normalized = scaler.fit_transform(df[col].values.reshape(-1, 1))
                df[col] = pd.Series(normalized.reshape(-1))
        df.to_csv(output_path+crypto,sep=",", index=False)

def standardization(input_path,output_path):
    folder_creator(output_path,1)
    excluded_features = ['Date']
    for crypto in os.listdir(input_path):
        df = pd.read_csv(input_path+crypto,delimiter=',', header=0)
        scaler = StandardScaler()
        for col in df.columns:
            if col not in excluded_features:
                normalized = scaler.fit_transform(df[col].values.reshape(-1, 1))
                df[col] = pd.Series(normalized.reshape(-1))
        df.to_csv(output_path+crypto,sep=",", index=False)

#creates the horizontal dataset
def create_horizontal_dataset(data_path,output_path):
    folder_creator(output_path+"horizontal_dataset/",1)
    dataframes=[]
    cryptocurrencies=os.listdir(data_path)
    cryptos_in_the_cluster=[]

    #take just the date column one time
    for crypto in cryptocurrencies:
        df_date=pd.read_csv(data_path+crypto,usecols=['Date'])
        dataframes.append(df_date)
        break

    # creates Close_1,Open_1 ecc for each dataframe
    i=1
    for crypto in os.listdir(data_path):
        df=pd.read_csv(data_path+crypto)
        cryptos_in_the_cluster.append(crypto.replace(".csv",""))
        df=df.drop('Date',axis=1)
        df=df.add_suffix('_'+str(i))
        i+=1
        dataframes.append(df)

    #concat horizontally all the dataframes
    horizontal = pd.concat(dataframes, axis=1)

    #serialization
    horizontal.to_csv(output_path+"horizontal_dataset/horizontal.csv",sep=",",index=False)

    return cryptos_in_the_cluster


