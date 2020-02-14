import os
import numpy as np
import pandas as pd
from crypto_utility.writer import save_distance_matrix, save_cryptoSymbol_id
from dtaidistance import dtw
from tqdm import tqdm
from dtaidistance import dtw_visualisation as dtwvis
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import minkowski
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.spatial.distance import euclidean

from distance_metrics import lcs
import sys

def generate_cryptocurrencies_dictionary(folderpreprocessing,filename):
    cryptocurrenciesSymbols = []
    for file in os.listdir(folderpreprocessing+"/step2_normalized/"):
        crypto = file.replace("_normalized.csv", "")
        cryptocurrenciesSymbols.append(crypto)
    length = len(cryptocurrenciesSymbols)
    rg = range(length)
    dictionary = dict(zip(rg, cryptocurrenciesSymbols))
    save_cryptoSymbol_id(dictionary,filename)
    return length,dictionary

def compute_distance_matrix(distance_measure,columnsToDrop,nameFileOutput,folderpreprocessing):
    length,dictionary= generate_cryptocurrencies_dictionary(folderpreprocessing,nameFileOutput)
    distance_matrix = np.zeros((length, length))
    for i in tqdm(range(length)):
    #for i in range(length):
        df = pd.read_csv(folderpreprocessing+"/step2_normalized/" + dictionary[i] + "_normalized.csv", delimiter=',',header=0)
        j=i+1
        while (j < length):
            """In questo punto l'obiettivo è di confrontare serie temporali che coprono lo stesso arco temporale"""
            dfTemp= df
            df1 = pd.read_csv(folderpreprocessing+"/step2_normalized/"+dictionary[j]+"_normalized.csv", delimiter=',',header=0)
            """print(dictionary[i] + ": "+ df['DateTime'][0])
            print(dictionary[j] + ": "+ df1['DateTime'][0])"""

            if(df['DateTime'][0] >= df1['DateTime'][0]):
                df1 = df1[df1['DateTime'] >= df['DateTime'][0]]
            else:
                dfTemp = df[df['DateTime'] >= df1['DateTime'][0]]

            """
            print(dictionary[i] + ": " + dfTemp['DateTime'])
            print(dictionary[j] + ": " + df1['DateTime'])"""

            df1 = df1.drop(columns=["DateTime",'Symbol'])
            dfTemp = dfTemp.drop(columns=["DateTime",'Symbol'])

            df1= df1.drop(columns=columnsToDrop)
            dfTemp = dfTemp.drop(columns=columnsToDrop)
            #print("Distanza tra [" + str(i) + "-" + dictionary[i] + "," + str(j) + "-" + dictionary[j] + "]: ")
            if (distance_measure=="dtw"):
                distances = []
                for col in df1.columns:
                    distance=dtw.distance(np.array(dfTemp[col].values).squeeze(), np.array(df1[col].values).squeeze())
                    distances.append(distance)
                ensemble_distance = np.average(distances)
                distance_matrix[i][j] = ensemble_distance
                distance_matrix[j][i] = ensemble_distance  # matrice simmetrica
            elif (distance_measure=="pearson"):
             distances=[]
             for col in df1.columns:
                 correlation, p = pearsonr(np.array(dfTemp[col].values).squeeze(), np.array(df1[col].values).squeeze())
                 distance = 1 - correlation  # lo trasformo in distanza: varia tra [0,2]
                 distances.append(distance)
             ensemble_distance= np.average(distances)
             distance_matrix[i][j] = ensemble_distance
             distance_matrix[j][i] = ensemble_distance #matrice simmetrica
             #print(ensemble_distance)
            elif(distance_measure=="euclidean"):
                distances = []
                for col in df1.columns:
                    distance1 = euclidean(np.array(dfTemp[col].values).squeeze(), np.array(df1[col].values).squeeze())
                    distance2 = dtw.distance(np.array(dfTemp[col].values).squeeze(), np.array(df1[col].values).squeeze())
                    print("euclidean "+str(distance1))
                    print("\n dtw: "+str(distance2))
            else:
                return "Distance measure is not valid"
            j+=1
    #salvo la matrice
    save_distance_matrix(distance_matrix,nameFileOutput)





