import csv
import json
import numpy as np
import pandas as pd
from utility.folder_creator import folder_creator
from utility.reader import get_dict_symbol_id


def save_clusters(clusters,filename,CLUSTERING_PATH):
    dict_symbol_id= get_dict_symbol_id(CLUSTERING_PATH)
    folder_creator(CLUSTERING_PATH+"clusters/", 0)
    df=pd.DataFrame(columns=['cluster_id','cryptos'])
    i=0
    for cluster in clusters:
        cryptocurrencies = []
        for crypto_id in cluster:
            cryptocurrencies.append(dict_symbol_id.symbol[crypto_id])
        df = df.append({'cluster_id': str(i), 'cryptos': cryptocurrencies}, ignore_index=True)
        i += 1
    df.to_csv(CLUSTERING_PATH+"clusters/"+filename+".csv",sep=",",index=False)

def save_distance_matrix(distance_matrix,FINAL_PATH):
    with open(FINAL_PATH+"distance_matrix.csv", 'w') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerows(distance_matrix)
    writeFile.close()

def save_dict_symbol_id(dict):
    with open('../modelling/techniques/clustering/symbol_id.json', 'w') as fp:
        json.dump(dict, fp)
    fp.close()
