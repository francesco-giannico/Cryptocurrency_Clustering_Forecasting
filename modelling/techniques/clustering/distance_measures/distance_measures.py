import os
import numpy as np
import pandas as pd
from dtaidistance import dtw
from tqdm import tqdm
from dtaidistance import dtw_visualisation as dtwvis
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import minkowski
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.spatial.distance import euclidean
import sys

from modelling.techniques.clustering.consensus_clustering import consensus_clustering
from modelling.techniques.clustering.distance_measures.coral_distance import CORAL
from utility.clustering_utils import generate_cryptocurrencies_dictionary
from utility.cut import cut_dataset_by_range
from utility.reader import get_dict_symbol_id
from utility.writer import save_distance_matrix
from scipy.stats import wasserstein_distance

PATH_NORMALIZED_FOLDER="../preparation/preprocessed_dataset/constructed/normalized/"

def main_clustering():
    #todo genero il nuovo crypto_symbols ?
    #generate_cryptocurrencies_dictionary()
    #todo read dict symbol
    #dict_symbol_id= get_dict_symbol_id()
    #todo genero la nuova distance matrix?
    #compute_distance_matrix(dict_symbol_id,"wasserstain")
    #consensus_clustering
    consensus_clustering("wasserstain")

def compute_distance_matrix(dict_symbol_id,distance_measure):
    dict_length = dict_symbol_id.symbol.count()
    distance_matrix = np.zeros((dict_length,dict_length))
    #for i in tqdm(range(length)):
    for i in range(dict_length):
        df = cut_dataset_by_range(PATH_NORMALIZED_FOLDER, dict_symbol_id.symbol[i],
                                  "2019-01-01", "2019-12-31")
        df = df.set_index("Date")
        j=i+1
        while (j < dict_length):
            df1 = cut_dataset_by_range(PATH_NORMALIZED_FOLDER, dict_symbol_id.symbol[j],
                                      "2019-01-01", "2019-12-31")
            print("working on "+ dict_symbol_id.symbol[i] + "-"+ dict_symbol_id.symbol[j])
            df1=df1.set_index("Date")
            if (distance_measure=="dtw"):
                distances = []
                for col in df.columns:
                    distance=dtw.distance(np.array(df[col].values).astype(np.float).squeeze(), np.array(df1[col].values).astype(np.float).squeeze())
                    distances.append(distance)
                ensemble_distance = np.average(distances)
                distance_matrix[i][j] = ensemble_distance
                distance_matrix[j][i] = ensemble_distance  # matrice simmetrica
            elif (distance_measure=="pearson"):
                 distances=[]
                 for col in df.columns:
                     correlation, p = pearsonr(df[col].to_numpy(dtype="float"),df1[col].to_numpy(dtype="float"))
                     distance = 1 - correlation  # lo trasformo in distanza: varia tra [0,2]
                     distances.append(distance)
                 #distanza tra tutte le colonne..
                 ensemble_distance= np.average(distances)
                 # matrice simmetrica
                 distance_matrix[i][j] = ensemble_distance
                 distance_matrix[j][i] = ensemble_distance
            elif(distance_measure=="euclidean"):
                distances = []
                for col in df.columns:
                    distance1 = euclidean(np.array(df[col].values).squeeze(), np.array(df[col].values).squeeze())
                    distance2 = dtw.distance(np.array(df[col].values).squeeze(), np.array(df[col].values).squeeze())
                    print("euclidean "+str(distance1))
                    print("\n dtw: "+str(distance2))
            elif (distance_measure == "wasserstain"):
                distances = []
                for col in df.columns:
                    distance = wasserstein_distance(df[col].to_numpy(dtype="float"),df1[col].to_numpy(dtype="float"))
                    distances.append(distance)
                ensemble_distance = np.average(distances)
                # matrice simmetrica
                distance_matrix[i][j] = ensemble_distance
                distance_matrix[j][i] = ensemble_distance
            elif (distance_measure == "coral"):
                coral = CORAL()
                distances = []
                for col in df.columns:
                    distance = coral.fit(df[col].to_numpy(dtype="float"),df1[col].to_numpy(dtype="float"))
                    distances.append(distance)
                    print(distance)
                ensemble_distance = np.average(distances)
                # matrice simmetrica
                distance_matrix[i][j] = ensemble_distance
                distance_matrix[j][i] = ensemble_distance
            else:
                return "Distance measure is not valid"
            j+=1
    #salvo la matrice
    save_distance_matrix(distance_matrix,distance_measure)





