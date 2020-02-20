import numpy as np
import pandas as pd
from dtaidistance import dtw
from scipy.stats import pearsonr
from scipy.spatial.distance import euclidean
from modelling.techniques.clustering.distance_measures.coral_distance import CORAL
from utility.dataset_utils import cut_dataset_by_range
from utility.writer import save_distance_matrix
from scipy.stats import wasserstein_distance


def compute_distance_matrix(dict_symbol_id,distance_measure,start_date,end_date,CLUSTERING_PATH):
    dict_length = dict_symbol_id.symbol.count()
    distance_matrix = np.zeros((dict_length,dict_length))
    for i in range(dict_length):
        df = pd.read_csv(CLUSTERING_PATH+"cut_datasets/"+dict_symbol_id.symbol[i]+".csv", sep=",",header=0)
        df = df.set_index("Date")
        j=i+1
        while (j < dict_length):
            df1 = pd.read_csv(CLUSTERING_PATH + "cut_datasets/" + dict_symbol_id.symbol[j]+".csv", sep=",", header=0)
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
    #save the matrix
    save_distance_matrix(distance_matrix,CLUSTERING_PATH)





