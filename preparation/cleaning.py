import os
import shutil
import pandas as pd
from scipy.stats import stats

from utility.dataset_utils import cut_dataset_by_range
from utility.folder_creator import folder_creator
import numpy as np
PATH_PREPROCESSED_FOLDER="../preparation/preprocessed_dataset/"
PATH_UNCOMPLETE_FOLDER="../preparation/preprocessed_dataset/selected/uncomplete/"
PATH_COMPLETE_FOLDER="../preparation/preprocessed_dataset/selected/complete/"
PATH_CLEANED_FOLDER="../preparation/preprocessed_dataset/cleaned/"

def remove_uncomplete_rows_by_range(crypto_symbol,start_date,end_date):
 folder_creator(PATH_CLEANED_FOLDER,0)
 folder_creator(PATH_CLEANED_FOLDER+"partial", 0)
 df=cut_dataset_by_range(PATH_UNCOMPLETE_FOLDER,crypto_symbol,start_date,end_date)
 df.to_csv(PATH_CLEANED_FOLDER+"partial/"+crypto_symbol+".csv",sep=",",index=False)

def input_missing_values():
    folder_creator(PATH_CLEANED_FOLDER+"final",1)
    #already_treated=['LKK.csv','FAIR.csv']
    """for crypto_symbol in os.listdir(PATH_CLEANED_FOLDER+"partial"):
        df = pd.read_csv(PATH_CLEANED_FOLDER+"partial/"+crypto_symbol, delimiter=',', header=0)
        already_treated.append(crypto_symbol)
        df=interpolate_with_time(df)
        df.to_csv(PATH_CLEANED_FOLDER + "final/" + crypto_symbol, sep=",", index=False)"""

    for crypto_symbol in os.listdir(PATH_UNCOMPLETE_FOLDER):
        df = pd.read_csv(PATH_UNCOMPLETE_FOLDER+crypto_symbol, delimiter=',', header=0)
         #if crypto_symbol not in already_treated:
        df=interpolate_with_time(df)
        df.to_csv(PATH_CLEANED_FOLDER + "final/" + crypto_symbol , sep=",", index=False)

    #merge with complete dataset
    for crypto_symbol in os.listdir(PATH_COMPLETE_FOLDER):
        shutil.copy(PATH_COMPLETE_FOLDER+ crypto_symbol, PATH_CLEANED_FOLDER+ "final/" + crypto_symbol)

#todo spiegare come mai hai scelto questo metodo di interpolazione... ce ne sono tanti a disposizione
def interpolate_with_time(df):
    # Converting the column to DateTime format
    df.Date = pd.to_datetime(df.Date, format='%Y-%m-%d')
    df = df.set_index('Date')
    # interpolate with time
    df = df.interpolate(method='time')
    df = df.reset_index()
    return df


PATH_COMPLETE_FOLDER="../preparation/preprocessed_dataset/selected/complete/"
def remove_outliers():
    for crypto in os.listdir(PATH_COMPLETE_FOLDER):
        df=pd.read_csv(PATH_COMPLETE_FOLDER+crypto,sep=",",header=0,usecols=['Date','Close'])
        close_mean=df.Close.mean()
        close_std=df.Close.std()
        low=0.30
        high=0.95
        res=df.Close.quantile([low,high])
        #print(res)
        true_index=(res.loc[low] < df.Close.values) & (df.Close.values < res.loc[high])
        #print(true_index)
        false_index=~true_index
        #df1=df[false_index]
        #print(df1.describe())
        #df = df[(np.abs(stats.zscore(df, axis=1)) < 3).all(axis=1)]
        """df = df[np.abs(df.Close - close_mean)  <= (2* close_std)]
        df = df[np.abs(df.Close - close_mean)  > (2* close_std)]"""
        df.Close=df.Close[true_index]
        df=df[true_index]
        #df.Close[false_index]=np.median(df.Close[true_index])
        #print(df.head())
        df.to_csv(PATH_CLEANED_FOLDER+"final/"+crypto,sep=",",index=False)
        #print(df.describe())