import numpy as np
from math import sqrt
from pyclustering.cluster.agglomerative import agglomerative, type_link
import pandas as pd
from pyclustering.cluster.kmedoids import kmedoids
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from utility.folder_creator import folder_creator
from utility.writer import save_clusters
from pyclustering.cluster.silhouette import silhouette_ksearch, silhouette_ksearch_type, silhouette
from pyclustering.cluster.xmeans import xmeans
from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from sklearn.metrics import silhouette_samples
from pyclustering.cluster.kmeans import kmeans, kmeans_visualizer
from pyclustering.samples.definitions import FCPS_SAMPLES
from pyclustering.utils import read_sample
import matplotlib.pyplot as plt


def consensus_clustering(CLUSTERING_PATH):
    df = pd.read_csv(CLUSTERING_PATH+"distance_matrix.csv", delimiter=',', header=None)
    sample = df.values
    #number of elements
    N= len(df.columns)

    #k=1
    #k_c = [k_sqrtN, k_sqrtNDiv2, k_sqrtNBy2, k_sqrtNDiv4, k_sqrtNBy4]
    #rule of thumbs for k
    df1= pd.DataFrame(columns=['value'],index=['k_sqrtN','k_sqrtNDiv2','k_sqrtNBy2','k_sqrtNDiv4','k_sqrtNBy4'])
    df1.at['k_sqrtN','value']= int(sqrt(N))
    df1.at['k_sqrtNDiv2', 'value'] = int(sqrt(N / 2))
    df1.at['k_sqrtNBy2', 'value'] = int(sqrt(N * 2))
    df1.at['k_sqrtNDiv4', 'value'] = int(sqrt(N / 4))
    df1.at['k_sqrtNBy4', 'value'] = int(sqrt(N * 4))

    # Declare the weight of each vote
    # consensus matrix is NxN
    iterations=20
    weight = 1 / iterations
    weight2 = 1 / len(df1.index)
    consensus_matrix = np.zeros((N, N))
    for k in df1.index:
        #avvio lo stesso algoritmo con k diversi, ogni volta 4 volte
        for iteration in range(iterations):
            initial_medoids = kmeans_plusplus_initializer(sample, df1.loc[k].values[0]).initialize(return_index=True)
            kmedoids_instance = kmedoids(sample, initial_medoids,data_type="distance_matrix")
            kmedoids_instance.process()
            clusters = kmedoids_instance.get_clusters()
            coassociations_matrix_new= np.zeros((N, N))
            for cluster in clusters:
                for crypto in cluster:
                    coassociations_matrix_new[crypto][crypto] = 1
                    for crypto1 in cluster:
                        coassociations_matrix_new[crypto][crypto1]= 1
                        coassociations_matrix_new[crypto1][crypto] = 1
            consensus_matrix=consensus_matrix+coassociations_matrix_new
    consensus_matrix = consensus_matrix*weight*weight2
        # adesso fai 1- consensus_matrix(migliore) e ottieni la dissimilarity/distance matrix da usare con K-means o agglomerative
    #print(consensus_matrix)
    distance_matrix= 1-consensus_matrix

    for k in df1.index:
        agglomerative_instance = agglomerative(distance_matrix, df1.loc[k].values[0], type_link.COMPLETE_LINK)
        agglomerative_instance.process()
        # Obtain results of clustering
        clusters = agglomerative_instance.get_clusters()
        save_clusters(clusters,k,CLUSTERING_PATH)