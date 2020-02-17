import shutil
import pandas as pd
import os
import math
import pandas as pd
from utility.reader import read_csv
from utility.folder_creator import folder_creator

""" it Moves the crypto dead before 31-12-2019 in the dead folder """
PATH_MAIN_FOLDER="../acquisition/dataset/original/"
PATH_LESS_FEATURES= "../preparation/preprocessed_dataset/selected/less_features/"
PATH_PREPARATION_FOLDER="../preparation/preprocessed_dataset/"

def find_by_dead_before():
    folder_creator(PATH_PREPARATION_FOLDER + "selected/"+"dead", 1)
    for file in os.listdir(PATH_LESS_FEATURES):
        df = pd.read_csv(PATH_LESS_FEATURES + file, delimiter=',', header=0)
        df = df.set_index("Date")
        # dead before
        lastDate = df.index[::-1][0]
        if lastDate != '2019-12-31':
            shutil.copy(PATH_LESS_FEATURES + file,PATH_PREPARATION_FOLDER+"selected/dead/" + file)

""" it moves the crypto with null values in the uncomplete folder """
def find_uncomplete():
   folder_creator(PATH_PREPARATION_FOLDER + "selected/" + "uncomplete", 1)
   folder_creator(PATH_PREPARATION_FOLDER + "selected/" + "complete", 1)
   #print(df.columns.values.tolist())
   for file in os.listdir(PATH_LESS_FEATURES):
    df = pd.read_csv(PATH_LESS_FEATURES+file, delimiter=',',header=0)
    df=df.set_index("Date")
    #with null values
    if(df["Open"].isnull().any()):
         try:
            shutil.copy(PATH_LESS_FEATURES+file,PATH_PREPARATION_FOLDER+"selected/uncomplete/"+file)
         except:
             pass
    else:
        try:
            shutil.copy(PATH_LESS_FEATURES + file, PATH_PREPARATION_FOLDER+ "selected/complete/" + file)
        except:
            pass

def remove_features(features_to_remove):
    folder_creator(PATH_PREPARATION_FOLDER+"selected/",1)
    folder_creator(PATH_PREPARATION_FOLDER + "selected/less_features", 1)
    for crypto in os.listdir(PATH_MAIN_FOLDER):
        df = pd.read_csv(PATH_MAIN_FOLDER + crypto, delimiter=',', header=0)
        for feature in features_to_remove:
            del df[feature]
            df.to_csv(PATH_PREPARATION_FOLDER+"selected/less_features/"+crypto,sep=",",index=False)



"""def get_date_crypto_less_entry(folderpreprocessing,cluster):
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
    return csv['DateTime'][0]"""