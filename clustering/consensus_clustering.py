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

path_distance_matrix="crypto_clustering/distanceMeasures/distanceMatrices/"
def consensus_clustering(k_n,distance_matrix,folderoutput):
    folder_creator("crypto_clustering/"+folderoutput+"/clusters", 0)
    df = pd.read_csv(path_distance_matrix+distance_matrix+".csv", delimiter=',', header=None)
    print(df.shape)
    sample = df.values

    N= len(df.columns)
    k=1
    k_sqrtN = int(sqrt(N))
    k_sqrtNDiv2 = int(sqrt(N / 2))
    k_sqrtNBy2 = int(sqrt(N * 2))
    k_sqrtNDiv4 = int(sqrt(N / 4))
    k_sqrtNBy4 = int(sqrt(N * 4))
    k_c = [k_sqrtN, k_sqrtNDiv2, k_sqrtNBy2, k_sqrtNDiv4, k_sqrtNBy4]

    # Declare the weight of each vote
    # consensus matrix is NxN

    iterations=20
    weigth = 1 / iterations
    weigth2 = 1 / len(k_c)
    consensus_matrix = np.zeros((N, N))
    for k in k_c:
        #avvio lo stesso algoritmo con k diversi, ogni volta 4 volte
        for iteration in range(iterations):
            initial_medoids = kmeans_plusplus_initializer(sample, k).initialize(return_index=True)
            kmedoids_instance = kmedoids(sample, initial_medoids,data_type="distance_matrix")
            kmedoids_instance.process()
            clusters=  kmedoids_instance.get_clusters()
            coassociations_matrix_new= np.zeros((N, N))
            for cluster in clusters:
                for crypto in cluster:
                    coassociations_matrix_new[crypto][crypto] = 1
                    for crypto1 in cluster:
                        coassociations_matrix_new[crypto][crypto1]= 1
                        coassociations_matrix_new[crypto1][crypto] = 1
            consensus_matrix=consensus_matrix+coassociations_matrix_new
    consensus_matrix = consensus_matrix*weigth*weigth2
        # adesso fai 1- consensus_matrix migliore e ottieni la dissimilarity/distance matrix da usare con K-means o agglomerative
    #print(consensus_matrix)
    distanceMatrix= 1-consensus_matrix

    i = 0
    for k in k_c:
        # Create object that uses python code only
        agglomerative_instance = agglomerative(distanceMatrix, k, type_link.COMPLETE_LINK)
        # Cluster analysis
        agglomerative_instance.process()
        # Obtain results of clustering
        clusters = agglomerative_instance.get_clusters()
        save_clusters(clusters, k_n[i],folderoutput)
        i+=1










