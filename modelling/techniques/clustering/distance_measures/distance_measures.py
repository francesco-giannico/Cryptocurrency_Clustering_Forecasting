import numpy as np
import pandas as pd
from dtaidistance import dtw
from scipy.stats import pearsonr
from scipy.spatial.distance import euclidean
from modelling.techniques.clustering.distance_measures.coral_distance import CORAL
from understanding.exploration import bivariate_plot
from utility.dataset_utils import cut_dataset_by_range
from utility.folder_creator import folder_creator
from utility.writer import save_distance_matrix
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt


def compute_distance_matrix(dict_symbol_id,distance_measure,CLUSTERING_PATH,features_to_use=None):
    dict_length = dict_symbol_id.symbol.count()
    distance_matrix = np.zeros((dict_length,dict_length))

    for i in range(dict_length):
        df = pd.read_csv(CLUSTERING_PATH+"cut_datasets/"+dict_symbol_id.symbol[i]+".csv", sep=",",header=0)
        df = df.set_index("Date")
        j=i+1
        while (j < dict_length):
            df1 = pd.read_csv(CLUSTERING_PATH + "cut_datasets/" + dict_symbol_id.symbol[j]+".csv", sep=",", header=0)
            print("distance between "+ dict_symbol_id.symbol[i] + "-"+ dict_symbol_id.symbol[j])
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
                 if features_to_use == None:
                     features = df.columns
                 else:
                     features = features_to_use
                 for col in features:
                     """
                     bivariate_plot(dfL,[col,col+"_2"])"""
                     if (dict_symbol_id.symbol[i] =="BTC"):
                         import seaborn as sns
                         dfL = pd.DataFrame()
                         dfL[col] = df1[col]
                         dfL[col + "_2"] = df[col]
                         # Set style of scatterplot
                         """sns.set_context("notebook", font_scale=1.1)
                         sns.set_style("ticks")"""

                         # Create scatterplot of dataframe
                         sns.lmplot(col,  # Horizontal axis
                                    col+"_2",  # Vertical axis
                                    data=dfL,  # Data source,
                                    line_kws={'color': 'red'})  # S marker size

                         # Set title
                         plt.title( dict_symbol_id.symbol[i] + "-"+ dict_symbol_id.symbol[j])

                         # Set x-axis label
                         plt.xlabel(col)
                         # Set y-axis label
                         plt.ylabel(col+"_2")
                         folder_creator(CLUSTERING_PATH+col,0)
                         plt.savefig(CLUSTERING_PATH+col+"/"+dict_symbol_id.symbol[i] + "-"+ dict_symbol_id.symbol[j]+".png")
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
                if features_to_use == None:
                    features = df.columns
                else:
                    features = features_to_use
                for col in features:
                    if col!="Date":
                        distance = wasserstein_distance(df[col].to_numpy(dtype="float"),df1[col].to_numpy(dtype="float"))
                        distances.append(distance)
                        #print(" column " + col + " value "+ str(distance))
                ensemble_distance = np.average(distances)
                # since the matrix is simmetric then:
                distance_matrix[i][j] = ensemble_distance
                distance_matrix[j][i] = ensemble_distance

            elif (distance_measure == "coral"):
                coral = CORAL()
                distances = []
                for col in df.columns:
                    distance = coral.fit(df[col].to_numpy(dtype="float"),df1[col].to_numpy(dtype="float"))
                    distances.append(distance)
                ensemble_distance = np.average(distances)
                # matrice simmetrica
                distance_matrix[i][j] = ensemble_distance
                distance_matrix[j][i] = ensemble_distance
            else:
                return "Distance measure unrecognized"
            j+=1

    #creating distance matrix heatmap
    dict_symbol_id=dict_symbol_id.reset_index()
    df3=pd.DataFrame(data=distance_matrix,columns=dict_symbol_id.symbol.values,index=dict_symbol_id.symbol.values)
    import seaborn as sns
    plt.figure(figsize=(20, 10))
    dist_chart=sns.heatmap(
        df3,
        cmap='OrRd',
        linewidth=1,
        annot=True,
    )
    dist_chart.set_xticklabels(dist_chart.get_xticklabels(), rotation=45, horizontalalignment='right')
    dist_chart.set_yticklabels(dist_chart.get_yticklabels(), rotation=10)
    dist_chart.set_title("Distance Matrix using "+distance_measure+" Distance")
    plt.savefig(CLUSTERING_PATH+"distance_matrix.png",dpi=120)
    #save the matrix
    save_distance_matrix(distance_matrix,CLUSTERING_PATH)





