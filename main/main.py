import os
from datetime import datetime
from itertools import product
from math import sqrt
import pandas as pd
from modelling.techniques.baseline.simple_prediction.simple_prediction import simple_prediction
from modelling.techniques.baseline.vector_autoregression.vector_autoregression import vector_autoregression
from modelling.techniques.clustering.clustering import clustering
from modelling.techniques.forecasting.multi_target import multi_target
from modelling.techniques.forecasting.single_target import single_target
from modelling.techniques.forecasting.testing.test_set import generate_testset, get_testset
from preparation.construction import create_horizontal_dataset
from preparation.preprocessing import preprocessing
from acquisition.yahoo_finance_history import get_most_important_cryptos
from understanding.exploration import missing_values, describe
from utility.clustering_utils import merge_predictions
from utility.dataset_utils import cut_dataset_by_range
from utility.folder_creator import folder_creator
from visualization.bar_chart.forecasting import report_configurations, report_crypto
from visualization.line_chart import generate_line_chart
import numpy as np

def main():
    #DATA UNDERSTANDING
    #data_understanding()

    #DATA PREPARATION
    #preprocessing()

    #CLUSTERING
    start_date = "2014-10-01"
    end_date = "2019-12-31"
    distance_measure = "wasserstain"
    #clustering_main(distance_measure,start_date,end_date)

    #TESTING SET
    TEST_SET=testing_set()

    #MODELLING
    #single_target_main(distance_measure,start_date,end_date,TEST_SET)
    #MULTITARGET
    multi_target_main(distance_measure,start_date,end_date,TEST_SET)

def data_understanding():
    #DATA UNDERSTANDING
    PATH_DATASET= "../acquisition/dataset/original/"

    #COLLECT INITIAL DATA
    #todo data collecting from yahoo finance
    #get_most_important_cryptos(startdate=datetime(2010, 1, 2),enddate=datetime(2020, 1, 1))
    # EXPLORE DATA
    #missing_values(PATH_DATASET)
    #describe dataframes
    describe(PATH_DATASET)

def clustering_main(distance_measure,start_date,end_date):
    # clustering
    clustering(distance_measure, start_date=start_date, end_date=end_date)

def testing_set():
    test_start_date="2019-01-01"
    test_end_date="2019-12-31"
    try:
        TEST_SET = get_testset(
            "../modelling/techniques/forecasting/testing/" + test_start_date + "_" + test_end_date + ".txt")
    except:
      # Test set HAS TO BE EQUAL AMONG ALL THE EXPERIMENTS!!!
      generate_testset(test_start_date, test_end_date,"../modelling/techniques/forecasting/testing/")
      TEST_SET = get_testset(
          "../modelling/techniques/forecasting/testing/" + test_start_date + "_" + test_end_date + ".txt")
    return TEST_SET

def single_target_main(distance_measure,start_date,end_date,TEST_SET):
    DATA_PATH = "../modelling/techniques/clustering/output/" + distance_measure + "/" + start_date + "_" + end_date + "/cut_dataset_oring/"
    # SIMPLE PREDICTION
    simple_prediction(DATA_PATH,TEST_SET)

    # SINGLE TARGET LSTM
    temporal_sequences = [30,100]
    number_neurons = [128,256]
    learning_rate = 0.001
    EXPERIMENT_PATH = "../modelling/techniques/forecasting/output/" + distance_measure + "/" + start_date + "_" + end_date + "/single_target/"
    TENSOR_DATA_PATH = EXPERIMENT_PATH + "tensor_data"
    single_target(EXPERIMENT_PATH=EXPERIMENT_PATH,
                  DATA_PATH=DATA_PATH,
                  TENSOR_DATA_PATH=TENSOR_DATA_PATH,
                  window_sequence=temporal_sequences,
                  list_num_neurons=number_neurons, learning_rate=learning_rate,
                  testing_set=TEST_SET
                  )

    # visualization single_target
    """report_configurations(temporal_sequence=temporal_sequences,num_neurons=number_neurons,
                          experiment_folder=EXPERIMENT_PATH,results_folder="result",
                          report_folder="report",output_filename="overall_report")"""

    report_crypto(experiment_folder=EXPERIMENT_PATH,result_folder="result",report_folder="report",output_filename="report")

    #generate_line_chart(EXPERIMENT_PATH,temporal_sequences,number_neurons)"""

def multi_target_main(distance_measure,start_date,end_date,TEST_SET):
    DATA_PATH = "../modelling/techniques/clustering/output/" + distance_measure + "/" + start_date + "_" + end_date + "/clusters/"
    EXPERIMENT_PATH = "../modelling/techniques/forecasting/output/" + distance_measure + "/" + start_date + "_" + end_date + "/multi_target/"
    # SINGLE TARGET LSTM
    temporal_sequences = [30, 100]
    number_neurons = [128, 256]
    learning_rate = 0.001
    # folder_creator(EXPERIMENT_PATH + "clusters", 0)
    # reads each k used (folders'name)
    for k_used in os.listdir(DATA_PATH):
        #todo remove this one
        if k_used=="k_sqrtNDiv2":
            folder_creator(EXPERIMENT_PATH + "clusters" + "/" + k_used, 0)
            for cluster in os.listdir(DATA_PATH + k_used):
                if cluster.startswith("cluster_"):
                    folder_creator(EXPERIMENT_PATH + "clusters" + "/" + k_used + "/" + cluster + "/", 0)
                    # generate horizontal dataset
                    # leggere le criptovalute in questo dataset.
                    cryptos_in_the_cluster=create_horizontal_dataset(DATA_PATH+k_used+"/"+cluster+"/",EXPERIMENT_PATH + "clusters" + "/" + k_used+"/"+cluster+"/")

                    # Baseline - VECTOR AUTOREGRESSION
                    OUTPUT_FOLDER_VAR="../modelling/techniques/baseline/vector_autoregression/output/" + distance_measure + "/" + start_date + "_" + end_date + "/clusters/" + k_used+"/"+cluster+"/"
                    #vector_autoregression(EXPERIMENT_PATH + "clusters" + "/" + k_used+"/"+cluster+"/"+"horizontal_dataset/horizontal.csv",TEST_SET, OUTPUT_FOLDER_VAR)

                    # LSTM
                    dim_last_layer = len(os.listdir(DATA_PATH + k_used + "/" + cluster + "/"))
                    multi_target(EXPERIMENT_PATH=EXPERIMENT_PATH + "clusters" + "/" + k_used+"/"+cluster+"/",
                                      DATA_PATH=EXPERIMENT_PATH + "clusters" + "/" + k_used+"/"+cluster+"/"+"horizontal_dataset/",
                                      TENSOR_DATA_PATH=EXPERIMENT_PATH + "clusters" + "/" + k_used+"/"+cluster+"/" "tensor_data/",
                                      window_sequence=temporal_sequences,
                                      list_num_neurons=number_neurons, learning_rate=learning_rate,
                                      dimension_last_layer=dim_last_layer,testing_set=TEST_SET,cryptos=cryptos_in_the_cluster)

                    """ report_configurations(exp_type="multi_target",temporal_sequence=temporal_sequences,num_neurons=number_neurons,
                                            experiment_folder=EXPERIMENT_PATH + "clusters" + "/" + k_used+"/"+cluster+"/",results_folder="result",
                                            report_folder="report",output_filename="overall_report")
                    """
                    # report_crypto(experiment_folder=EXPERIMENT_PATH + "clusters" + "/" + k_used+"/"+cluster+"/",result_folder="result",report_folder="report",output_filename="report")

                    # generate_line_chart(EXPERIMENT_PATH + "clusters" + "/" + k_used+"/"+cluster+"/",temporal_sequences,number_neurons)

                    # other charts for clustering

main()