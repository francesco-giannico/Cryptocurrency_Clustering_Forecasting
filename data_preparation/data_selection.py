import shutil
import pandas as pd
import os
import math

COLUMNS=['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
"""
it separates files starting from the original dataset, 
the ones with null values are moved in with_null_values folder and the ones
dead before 31-12-2019 are moved in dead_before folder"""
def separate_files():
   #print(df.columns.values.tolist())
   for file in os.listdir("../data_acquisition/dataset/original/"):
    df = pd.read_csv("../dataset/original/"+file, delimiter=',',header=0)
    df=df.set_index("Date")

    #dead before
    lastDate=df.index[::-1][0]
    if lastDate!='2019-12-31':
        shutil.move("../dataset/original/" + file, "../dataset/dead_before/" + file)

    #with null values
    for column in COLUMNS:
      if(df[column].isnull().any()):
         #print(file)
         try:
            shutil.move("../dataset/original/"+file, "../dataset/with_null_values/"+file)
         except:
             pass
         break


def find_minimum_date():
    for file in os.listdir("../data_acquisition/dataset/with_null_values/"):
        df = pd.read_csv("../dataset/with_null_values/" + file, delimiter=',', header=0)
        df = df.set_index("Date")

        init_date=df.index[0]
        for row in df.itertuples():
            #print(row.Open)
            if (math.isnan(row.Open)):
                fin_date=row.Index
                #df=df.drop(df.index[init_date:row.Index])
                df=df.query('index < @init_date or index > @fin_date')
                init_date=row.Index
        df.to_csv('../dataset/reviewed/'+file,",")

import pandas as pd
from utility.reader import read_csv

def get_date_crypto_less_entry(folderpreprocessing,cluster):
    min=10000
    cryptoMin=''
    #dateTimeMin=pd.datetime.now()
    for id in cluster:
     for crypto in cluster[id]:
        csv= read_csv(folderpreprocessing+"/step2_normalized/"+ crypto +"_normalized.csv")
       # print(crypto+ " " +str(len(csv['Open'])))
        #ora conto quante entry ha il file
        length= len(csv['Open']) #conto le righe
        if length<=min:
           # dateTime=pd.to_datetime(csv["DateTime"][0])
            #if(dateTime<dateTimeMin):
            min = length
            #dateTimeMin = dateTime
            cryptoMin=crypto
       # print("cluster "+ str(id) + " la minima e': " + cryptoMin)
        #recupero la data piu' vecchia della criptovaluta con meno entry
    csv = read_csv(folderpreprocessing+"/step2_normalized/" + cryptoMin + "_normalized.csv")
    return csv['DateTime'][0]