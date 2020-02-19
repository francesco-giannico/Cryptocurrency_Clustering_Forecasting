import json
import os
import pandas as pd

fileDir = os.path.dirname(os.path.realpath(__file__))#
def get_preprocessed_crypto_symbols():
    crypto_symbols = []
    for file in os.listdir("../preparation/preprocessed_dataset/integrated/"):
        crypto = file.replace(".csv","")
        crypto_symbols.append(crypto)
    return crypto_symbols

def get_original_crypto_symbols():
    crypto_symbols= []
    fileToRead = read_file("../acquisition/crypto_symbols.txt")
    for line in fileToRead:
        crypto_symbols.append(line.replace("\n", ""))
    fileToRead.close()
    return crypto_symbols

def get_dict_symbol_id():
    df=pd.read_csv('../modelling/techniques/clustering/symbol_id.csv',sep=",",header=0,index_col=1)
    return df

"""def get_clusters(filename):
    return read_json('crypto_clustering/results/'+filename+".json")

def get_clusters2(path_experiment,filename):
    return read_json('crypto_clustering/'+path_experiment+"/clusters/"+filename+".json")"""

def read_file(path):
    return open(path, "r")

def read_csv(path):
    return pd.read_csv(path)
