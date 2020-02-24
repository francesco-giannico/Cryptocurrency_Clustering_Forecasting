from datetime import datetime
from math import sqrt
import pandas as pd
from modelling.techniques.clustering.clustering import clustering
from modelling.techniques.forecasting.single_target import single_target, single_target1
from modelling.techniques.forecasting.testing.test_set import generate_testset, get_testset
from preparation.preprocessing import preprocessing
from acquisition.yahoo_finance_history import get_most_important_cryptos
from understanding.exploration import missing_values
from utility.dataset_utils import cut_dataset_by_range


def main():
    #DATA UNDERSTANDING
    PATH_DATASET= "../acquisition/dataset/original/"

    #COLLECT INITIAL DATA
    #todo data collecting from yahoo finance
    #get_most_important_cryptos(startdate=datetime(2010, 1, 2),enddate=datetime(2020, 1, 1))
    # EXPLORE DATA
    # todo dataset exploration
    #missing_values(PATH_DATASET)

    #DATA PREPARATION
    #preprocessing("t")

    #clustering
    # todo calcolo la matrice delle distanze
    distance_measure="wasserstain"
    start_date = "2018-01-01"
    end_date = "2019-12-31"
    #clustering("wasserstain",start_date="2018-01-01",end_date="2019-12-31")

    #forecasting

    # GENERATING TESTING SET
    test_start_date = "2019-07-01"
    test_end_date = "2019-12-31"
    #generate_testset(test_start_date, test_end_date,"../modelling/techniques/forecasting/testing/")
    #READING TEST SET
    TEST_SET=get_testset("../modelling/techniques/forecasting/testing/"+test_start_date+"_"+test_end_date+".txt")
    temporal_sequence_considered = [30,100,200]
    number_neurons_LSTM = [128,256]
    learning_rate = 0.001
    EXPERIMENT_PATH="../modelling/techniques/forecasting/output/"+distance_measure+"/"+start_date+"_"+end_date+"/single_target/"
    DATA_PATH="../modelling/techniques/clustering/output/wasserstain/"+start_date+"_"+end_date+"/cut_datasets/"
    TENSOR_DATA_PATH=EXPERIMENT_PATH+"tensor_data"
    single_target1(EXPERIMENT_PATH=EXPERIMENT_PATH,
                  DATA_PATH=DATA_PATH,
                  TENSOR_DATA_PATH=TENSOR_DATA_PATH,
                  window_sequence=temporal_sequence_considered,
                  num_neurons=number_neurons_LSTM, learning_rate=learning_rate,
                  testing_set=TEST_SET
                  )
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