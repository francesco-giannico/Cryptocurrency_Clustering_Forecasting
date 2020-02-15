from datetime import datetime

from data_preparation.preprocessing import preprocessing
from data_acquisition.yahoo_finance_history import get_most_important_cryptos
from data_understanding.data_exploration import missing_values


def main():
    #DATA UNDERSTANDING
    PATH_DATASET="../data_acquisition/dataset/original/"
    #COLLECT INITIAL DATA
    #todo data collecting from yahoo finance
    #get_most_important_cryptos(startdate=datetime(2010, 1, 2),enddate=datetime(2020, 1, 1))
    # EXPLORE DATA
    # todo dataset exploration
    #missing_values(PATH_DATASET)
    missing_values(PATH_DATASET)
    #DATA PREPARATION
    #SELECT DATA (row selection e feature selection)
    #preprocessing("t")
    #CLEAN DATA
    #
    #todo data_preparation

    #todo calcolo la matrice delle distanze
    #todo dynamic time warping
    #compute_distance_matrix("dtw",[],"dtw_allFeatures_noindicators")
    #compute_distance_matrix("dtw", ['High','Low','Open'], "dtw_close_noindicators")
    #compute_distance_matrix("dtw", ['High','Low'], "dtw_openclose_noindicators")

    #compute_distance_matrix("dtw", ['High','Low','Open'], "dtw_close_indicators",folderpreprocessing=folderpreprocessing)
    #compute_distance_matrix("dtw", [], "dtw_allFeatures_indicators",folderpreprocessing=folderpreprocessing)
    #compute_distance_matrix("dtw", ['High','Low'], "dtw_closeopen_indicators")

    #todo pearson
    #compute_distance_matrix("pearson", [], "pearson_allFeatures_noindicators")

    #compute_distance_matrix("pearson", ['High','Low','Open'], "pearson_close_noindicators")
    #compute_distance_matrix("pearson", ['High','Low'], "pearson_closeopen_noindicators")

    #compute_distance_matrix("pearson", [], "pearson_allFeatures_indicators",folderpreprocessing=folderpreprocessing)
    #compute_distance_matrix("pearson", ['High','Low','Open'], "pearson_close_indicators",folderpreprocessing=folderpreprocessing)
    # compute_distance_matrix("pearson", ['High','Low'], "pearson_closeopen_indicators")


    k_n_consensus = ["consensus_k_sqrtN", "consensus_k_sqrtNdiv2", "consensus_k_sqrtNby2", "consensus_k_sqrtNdiv4", "consensus_k_sqrtNby4"]
    #k_n_medoids = ["kmedoids_k_sqrtN", "kmedoids_k_sqrtNdiv2", "kmedoids_k_sqrtNby2", "kmedoids_k_sqrtNdiv4","kmedoids_k_sqrtNby4"]

    """nameFolderoutput="experiment_pearson_close_indicators"
    folder_creator("crypto_clustering/"+nameFolderoutput,0)
    generate_clusters(k_n_consensus,"pearson_close_indicators",folderoutput=nameFolderoutput)
    experiment(k_n_consensus,nameFolderoutput,"crypto_preprocessing_indicators")"""


    #multi on all
    """nameFolderoutput = "experiment_common_indicators"
    folder_creator("crypto_clustering/" + nameFolderoutput, 0)
    save_dummy_clusters("noclustering_indicators_multitarget",nameFolderoutput)
    experiment(k_n_consensus, nameFolderoutput, "crypto_preprocessing_indicators")"""
    #experiment_single() #RICORDA DI METTERE "INDEXES!!!"
    #results()
    #compute_distance_matrix("euclidean", ['High','Low','Open'], "euclideantest",folderpreprocessing="crypto_preprocessing_noindicators")

    #generate_report()

main()