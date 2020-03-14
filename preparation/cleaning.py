import os
import shutil
import pandas as pd
from scipy.stats import stats
from sklearn.preprocessing import MinMaxScaler

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
def remove_outliers_one():
    for crypto in os.listdir(PATH_COMPLETE_FOLDER):
        df=pd.read_csv(PATH_COMPLETE_FOLDER+crypto,sep=",",header=0)
        #df=cut_dataset_by_range(PATH_COMPLETE_FOLDER,crypto.replace(".csv",""),'2019-01-01','2019-12-31')
        folder_creator(PATH_CLEANED_FOLDER+"final/",1)
        #df.to_csv(PATH_CLEANED_FOLDER + "final/" + crypto, sep=",", index=False)

        low=0.15
        high=0.95
        res=df.Close.quantile([low,high])
        print(res)
        true_index=(res.loc[low] < df.Close.values) & (df.Close.values < res.loc[high])
        false_index=~true_index
        #df.Close=df.Close[true_index]
        df.Close[false_index]=np.median(df.Close[true_index])
        #print(df.head())
        df[true_index].to_csv(PATH_CLEANED_FOLDER+"final/"+crypto,sep=",",index=False)
        df=df[true_index]
        break

from sklearn.cluster import DBSCAN

#usa complete folder (open,high,low and close)
def remove_outliers_dbscan():
    excluded_features = ['Date']
    for crypto in os.listdir(PATH_COMPLETE_FOLDER):
        #uses all features
        df=pd.read_csv(PATH_COMPLETE_FOLDER+crypto,sep=",",header=0)
        scaler = MinMaxScaler()
        for col in df.columns:
            if col not in excluded_features:
                normalized = scaler.fit_transform(df[col].values.reshape(-1, 1))
                df[col] = pd.Series(normalized.reshape(-1))

        model = DBSCAN(eps=0.2, min_samples=100).fit(df.drop('Date',axis=1))
        print (len(df[model.labels_==-1].values))
        #outliers
        #print(df[model.labels_==-1])

        #saving the not normalized one
        df = pd.read_csv(PATH_COMPLETE_FOLDER + crypto, sep=",", header=0)
        df.Close[model.labels_ == -1]=np.median(df.Close[model.labels_ != -1])
        df.Open[model.labels_ == -1] = np.median(df.Open[model.labels_ != -1])
        df.High[model.labels_ == -1] = np.median(df.High[model.labels_ != -1])
        df.Low[model.labels_ == -1] = np.median(df.Low[model.labels_ != -1])
        #print(df[model.labels_==-1].Close)
        df.to_csv(PATH_CLEANED_FOLDER+"final/"+crypto,sep=",",index=False)
        break

