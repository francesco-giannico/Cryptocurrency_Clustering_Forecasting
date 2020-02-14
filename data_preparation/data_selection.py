import shutil
import pandas as pd
import os
import math

from utility.folder_creator import folder_creator

""" it Moves the crypto dead before 31-12-2019 in the dead folder """

PATH_MAIN_FOLDER="../data_acquisition/dataset/"
def find_by_dead_before():
    folder_creator(PATH_MAIN_FOLDER+"selected",1)
    folder_creator(PATH_MAIN_FOLDER + "selected/"+"dead", 1)
    for file in os.listdir(PATH_MAIN_FOLDER+"original/"):
        df = pd.read_csv(PATH_MAIN_FOLDER+"original/" + file, delimiter=',', header=0)
        df = df.set_index("Date")
        # dead before
        lastDate = df.index[::-1][0]
        if lastDate != '2019-12-31':
            shutil.copy(PATH_MAIN_FOLDER+"original/" + file, PATH_MAIN_FOLDER+"selected/dead/" + file)

""" it moves the crypto with null values in the uncomplete folder """
def find_uncomplete():
   folder_creator(PATH_MAIN_FOLDER + "selected/" + "uncomplete", 1)
   folder_creator(PATH_MAIN_FOLDER + "selected/" + "complete", 1)
   #print(df.columns.values.tolist())
   for file in os.listdir(PATH_MAIN_FOLDER+"original/"):
    df = pd.read_csv(PATH_MAIN_FOLDER+"original/"+file, delimiter=',',header=0)
    df=df.set_index("Date")
    #with null values
    if(df["Open"].isnull().any()):
         try:
            shutil.copy(PATH_MAIN_FOLDER+"original/"+file, PATH_MAIN_FOLDER+"selected/uncomplete/"+file)
         except:
             pass
    else:
        try:
            shutil.copy(PATH_MAIN_FOLDER + "original/" + file, PATH_MAIN_FOLDER + "selected/complete/" + file)
        except:
            pass


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