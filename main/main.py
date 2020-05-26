import os
from datetime import datetime
from decimal import Decimal
from itertools import product
from math import sqrt
import pandas as pd
from modelling.techniques.baseline.simple_prediction.simple_prediction import simple_prediction
from modelling.techniques.baseline.vector_autoregression.vector_autoregression import vector_autoregression
from modelling.techniques.clustering.clustering import clustering
from modelling.techniques.clustering.visualization import describe_new
from modelling.techniques.forecasting.multi_target import multi_target
from modelling.techniques.forecasting.single_target import single_target
from modelling.techniques.forecasting.testing.test_set import generate_testset, get_testset, \
    generate_testset_baseline
from preparation.construction import create_horizontal_dataset
from preparation.preprocessing import preprocessing
from acquisition.yahoo_finance_history import get_most_important_cryptos
from understanding.exploration import describe
from utility.clustering_utils import merge_predictions
from utility.dataset_utils import cut_dataset_by_range
from utility.folder_creator import folder_creator
from visualization.bar_chart.clustering import compare_multi_baseline_single_target, crypto_oriented, \
    crypto_cluster_oriented, multi_vs_single
from visualization.bar_chart.forecasting import report_configurations, report_crypto
from visualization.line_chart import generate_line_chart
import numpy as np


def main():
    #DATA UNDERSTANDING
    #data_understanding()

    # TESTING SET
    TEST_SET = testing_set()
    start_date="2015-09-01"
    end_date_for_clustering="2018-12-31"
    #DATA PREPARATION
    #preprocessing(TEST_SET,start_date,end_date_for_clustering)

    # CLUSTERING
    distance_measure = "pearson"
    features_to_use = ['Close']
    type_clustering = "min_max_normalized"
    # Description after
    type = "max_abs_normalized"
    # clustering
    """clustering(distance_measure, type_for_clustering=type_clustering, type_for_prediction=type,
               features_to_use=features_to_use)"""

    """describe(PATH_DATASET="../preparation/preprocessed_dataset/constructed/"+type+"/",
             output_path="../preparation/preprocessed_dataset/",
             name_folder_res=type)"""

    #MODELLING
    features_to_use=['Date', 'Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume', 'VWAP',
       'SMA_14', 'SMA_21', 'SMA_5', 'SMA_12', 'SMA_26', 'SMA_13', 'SMA_30',
       'SMA_20', 'SMA_50', 'SMA_100', 'SMA_200', 'EMA_14', 'EMA_21', 'EMA_5',
       'EMA_12', 'EMA_26', 'EMA_13', 'EMA_30', 'EMA_20', 'EMA_50', 'EMA_100',
       'EMA_200', 'RSI_14', 'RSI_21', 'RSI_5', 'RSI_12', 'RSI_26', 'RSI_13',
       'RSI_30', 'RSI_20', 'RSI_50', 'RSI_100', 'RSI_200', 'MACD_12_26_9',
       'MACDH_12_26_9', 'MACDS_12_26_9', 'BBL_20', 'BBM_20', 'BBU_20', 'MOM', 'CMO',  'UO',
       'lag_1','trend']
    #features_to_use = ['Date','Close', 'trend']
    # General parameters
    temporal_sequences = [3]
    list_number_neurons = [10]
    learning_rate = 0.001
    DROPOUT = 0.45
    EPOCHS = 1
    PATIENCE= 1
    BATCH_SIZE=None
    single_target_main(TEST_SET,type,features_to_use,
                       temporal_sequences,list_number_neurons,learning_rate,DROPOUT,
                        EPOCHS,PATIENCE,BATCH_SIZE)
    #MULTITARGET
    temporal_sequences = [3]
    list_number_neurons = [10]
    learning_rate = 0.001
    DROPOUT = 0.45
    EPOCHS = 1
    PATIENCE = 1
    BATCH_SIZE=None
    cluster_n="cluster"
    features_to_use = [ 'Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume', 'VWAP',
                       'SMA_14', 'SMA_21', 'SMA_5', 'SMA_12', 'SMA_26', 'SMA_13', 'SMA_30',
                       'SMA_20', 'SMA_50', 'SMA_100', 'SMA_200', 'EMA_14', 'EMA_21', 'EMA_5',
                       'EMA_12', 'EMA_26', 'EMA_13', 'EMA_30', 'EMA_20', 'EMA_50', 'EMA_100',
                       'EMA_200', 'RSI_14', 'RSI_21', 'RSI_5', 'RSI_12', 'RSI_26', 'RSI_13',
                       'RSI_30', 'RSI_20', 'RSI_50', 'RSI_100', 'RSI_200', 'MACD_12_26_9',
                       'MACDH_12_26_9', 'MACDS_12_26_9', 'BBL_20', 'BBM_20', 'BBU_20', 'MOM', 'CMO', 'UO',
                        'trend','symbol']
    #features_to_use = [ 'Open', 'High', 'Close', 'trend','symbol']
    print(TEST_SET)
    """multi_target_main(TEST_SET,features_to_use,temporal_sequences,list_number_neurons,learning_rate,DROPOUT,EPOCHS,PATIENCE,
                      cluster_n,BATCH_SIZE)"""
    """describe_new(PATH_DATASET="../modelling/techniques/clustering/",
             output_path="../modelling/techniques/clustering/",
             name_folder_res=type)
    """
    #[df=pd.read_csv("modelling/techniques/clustering/output_to_use/clusters/cluster_1/ETH.csv",header=0)
    """df = pd.read_csv("ETH.csv", header=0)

    print(df.isnull().sum())"""
    # types=["k_1","k_sqrtN","k_sqrtNdiv2","k_sqrtNdiv4","k_sqrtNby2","k_sqrtNby4"]
    """types=['outputs_multi']
    single_target="outputs_single"
    for current in types:
        #path_baseline = "../modelling/techniques/baseline/simple_prediction/output/average_rmse/"
        path_baseline = "../modelling/techniques/baseline/simple_prediction/output/average_accuracy/"
        path_single_target = "../modelling/techniques/forecasting/"+single_target+"/single_target/result/"
        path_multi_target = "../modelling/techniques/forecasting/"+current+"/multi_target/clusters/"
        output_path="../modelling/techniques/forecasting/"+current+"/reports/"
        compare_multi_baseline_single_target(path_baseline, path_single_target, path_multi_target,output_path)"""

    """path_multi_target = "../modelling/techniques/forecasting/"
    crypto_oriented(path_multi_target,types)"""

    #NEW: cluster oriented
    """path_multitarget="../modelling/techniques/forecasting/clusters/"
    path_singletarget = "../modelling/techniques/forecasting/output_single/result/"
    path_output="../modelling/techniques/forecasting/reports/"
    crypto_cluster_oriented(path_multitarget,path_singletarget,path_output)"""


    #NEW 2
    """path_multi_target = "../modelling/techniques/forecasting/"
    multi_vs_single(path_multi_target, types)"""

def multi_target_main(TEST_SET,features_to_use, temporal_sequences, list_number_neurons, learning_rate,
                      DROPOUT, EPOCHS, PATIENCE,cluster_n,BATCH_SIZE=None):
    DATA_PATH = "../modelling/techniques/clustering/output_to_use/clusters/"
    EXPERIMENT_PATH = "../modelling/techniques/forecasting/outputs/multi_target/"
    folder_creator(EXPERIMENT_PATH + "clusters" + "/" + cluster_n + "/", 0)
    # generate horizontal dataset
    cryptos_in_cluster=create_horizontal_dataset(DATA_PATH + cluster_n + "/", EXPERIMENT_PATH + "clusters" + "/" + cluster_n + "/",TEST_SET)

    features_to_use_multi = ['Date']
    for i in range(len(cryptos_in_cluster)):
        for feature in features_to_use:
            features_to_use_multi.append(feature + "_" + str(i + 1))

    multi_target(EXPERIMENT_PATH=EXPERIMENT_PATH + "clusters" + "/" + cluster_n + "/",
                 DATA_PATH=EXPERIMENT_PATH + "clusters" + "/" + cluster_n + "/" + "horizontal_datasets/",
                 TENSOR_DATA_PATH=EXPERIMENT_PATH +"clusters" + "/" + cluster_n + "/tensor_data/",
                 window_sequences=temporal_sequences,
                 list_num_neurons=list_number_neurons, learning_rate=learning_rate,
                 cryptos=cryptos_in_cluster,
                 features_to_use=features_to_use_multi,
                 DROPOUT=DROPOUT, EPOCHS=EPOCHS, PATIENCE=PATIENCE,BATCH_SIZE=BATCH_SIZE,test_set=TEST_SET)

    """report_configurations(temporal_sequence=temporal_sequences, num_neurons=list_number_neurons,
                          experiment_folder=EXPERIMENT_PATH + "clusters" + "/" + cluster_n + "/",
                          results_folder="result",
                          report_folder="report", output_filename="overall_report")"""

    """report_crypto(experiment_folder=EXPERIMENT_PATH + "clusters" + "/" + cluster_n + "/",
                  result_folder="result",
                  report_folder="report",output_filename="report")"""

    # generate_line_chart(EXPERIMENT_PATH + "clusters" + "/" + k_used+"/"+cluster+"/",temporal_sequences,number_neurons)



def single_target_main(TEST_SET,type, features_to_use, temporal_sequences, number_neurons,
                       learning_rate, DROPOUT, EPOCHS, PATIENCE,BATCH_SIZE):
    DATA_PATH = "../preparation/preprocessed_dataset/constructed/" + type + "/"

    # SIMPLE PREDICTION
    OUTPUT_SIMPLE_PREDICTION = "../modelling/techniques/baseline/simple_prediction/output/"
    simple_prediction(DATA_PATH, TEST_SET, OUTPUT_SIMPLE_PREDICTION)

    # SINGLE TARGET LSTM
    EXPERIMENT_PATH = "../modelling/techniques/forecasting/outputs/single_target/"
    TENSOR_DATA_PATH = EXPERIMENT_PATH + "tensor_data"

    """single_target(EXPERIMENT_PATH=EXPERIMENT_PATH,
                  DATA_PATH=DATA_PATH,
                  TENSOR_DATA_PATH=TENSOR_DATA_PATH,
                  window_sequences=temporal_sequences,
                  list_num_neurons=number_neurons, learning_rate=learning_rate,
                  features_to_use=features_to_use,
                  DROPOUT=DROPOUT, EPOCHS=EPOCHS, PATIENCE=PATIENCE,BATCH_SIZE=BATCH_SIZE,test_set=TEST_SET)"""

    # visualization single_target
    """report_configurations(temporal_sequence=temporal_sequences, num_neurons=number_neurons,
                          experiment_folder=EXPERIMENT_PATH, results_folder="result",
                          report_folder="report", output_filename="overall_report")"""

    """report_crypto(experiment_folder=EXPERIMENT_PATH, result_folder="result", report_folder="report",
                 output_filename="report")"""
    #generate_line_chart(EXPERIMENT_PATH, temporal_sequences, number_neurons)


def data_understanding(crypto_names=None):
    # DATA UNDERSTANDING
    PATH_DATASET = "../acquisition/dataset/original/"

    # COLLECT INITIAL DATA
    # data collecting from yahoo finance
    # get_most_important_cryptos(crypto_names,startdate=datetime(2010, 1, 2),enddate=datetime(2020, 1, 1))

    # EXPLORE DATA
    # missing_values(PATH_DATASET)
    # describe dataframes
    OUTPUT_PATH = "../understanding/output/"
    describe(PATH_DATASET, OUTPUT_PATH, None, None)

def testing_set_baseline(interval):
    test_start_date = "2019-01-01"
    test_end_date = "2019-12-31"
    try:
        TEST_SET = get_testset(
            "../modelling/techniques/forecasting/testing/" + test_start_date + "_" + test_end_date +"_" + str(interval)+"_baseline.txt")
    except:
        generate_testset_baseline(test_start_date, test_end_date,interval, "../modelling/techniques/forecasting/testing/")
        TEST_SET = get_testset(
            "../modelling/techniques/forecasting/testing/" + test_start_date + "_" + test_end_date+"_"+ interval+"_baseline.txt")
    return TEST_SET

def testing_set():
    test_start_date = "2019-01-01"
    test_end_date = "2019-12-31"
    try:
        TEST_SET = get_testset(
            "../modelling/techniques/forecasting/testing/" + test_start_date + "_" + test_end_date + ".txt")
    except:
        # Test set HAS TO BE EQUAL AMONG ALL THE EXPERIMENTS!!!
        generate_testset(test_start_date, test_end_date, "../modelling/techniques/forecasting/testing/")
        TEST_SET = get_testset(
            "../modelling/techniques/forecasting/testing/" + test_start_date + "_" + test_end_date + ".txt")
    return TEST_SET


main()