import json
import os

import pandas as pd


def get_crypto_symbols():
    print(os.getcwd())
    crypto_symbols= []
    fileToRead = read_file("../data_acquisition/crypto_symbols.txt")
    for line in fileToRead:
        crypto_symbols.append(line.replace("\n", ""))
    fileToRead.close()
    return crypto_symbols

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
