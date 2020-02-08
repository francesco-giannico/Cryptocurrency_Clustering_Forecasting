import json

import pandas as pd


def get_cryptocurrenciesSymbols():
    cryptocurrenciesSymbols = []
    fileToRead = read_file("crypto_data/dataset/Crypto.txt")
    for line in fileToRead:
        cryptocurrenciesSymbols.append(line.replace("\n", "").replace("*", ""))
    fileToRead.close()
    return cryptocurrenciesSymbols

def read_json(path):
    with open(path, 'r') as f:
        result = json.load(f)
    f.close()
    return result

def get_clusters(filename):
    return read_json('crypto_clustering/results/'+filename+".json")

def get_clusters2(path_experiment,filename):
    return read_json('crypto_clustering/'+path_experiment+"/clusters/"+filename+".json")

def read_file(path):
    return open(path, "r")

def read_csv(path):
    return pd.read_csv(path)
