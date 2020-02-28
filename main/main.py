from datetime import datetime
from math import sqrt
import pandas as pd

from modelling.techniques.baseline.simple_prediction.simple_prediction import simple_prediction
from modelling.techniques.baseline.vector_autoregression.vector_autoregression import vector_autoregression
from modelling.techniques.clustering.clustering import clustering
from modelling.techniques.forecasting.single_target import single_target
from modelling.techniques.forecasting.testing.test_set import generate_testset, get_testset
from preparation.construction import create_horizontal_dataset
from preparation.preprocessing import preprocessing
from acquisition.yahoo_finance_history import get_most_important_cryptos
from understanding.exploration import missing_values
from utility.clustering_utils import merge_predictions
from utility.dataset_utils import cut_dataset_by_range
from visualization.bar_chart.forecasting import report_configurations, report_crypto
from visualization.line_chart import generate_line_chart


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
    # todo compute distance matrix
    distance_measure="wasserstain"
    start_date = "2018-01-01"
    end_date = "2019-12-31"
    #clustering("wasserstain",start_date="2018-01-01",end_date="2019-12-31")

    #forecasting

    # GENERATING TESTING SET
    test_start_date = "2019-07-01"
    test_end_date = "2019-12-31"
    #todo THE TEST SET HAS TO BE EQUAL AMONG ALL THE EXPERIMENTS!!!
    #generate_testset(test_start_date, test_end_date,"../modelling/techniques/forecasting/testing/")
    #READING TEST SET
    TEST_SET=get_testset("../modelling/techniques/forecasting/testing/"+test_start_date+"_"+test_end_date+".txt")


    #MODELLING

    EXPERIMENT_PATH="../modelling/techniques/forecasting/output/"+distance_measure+"/"+start_date+"_"+end_date+"/single_target/"
    DATA_PATH="../modelling/techniques/clustering/output/"+distance_measure+"/"+start_date+"_"+end_date+"/cut_datasets/"
    #DATA_PATH="../preparation/preprocessed_dataset/integrated/"

    #SIMPLE PREDICTION
    simple_prediction(DATA_PATH,TEST_SET)

    #SINGLE TARGET LSTM
    temporal_sequences = [30]
    number_neurons = [128]
    learning_rate = 0.001
    TENSOR_DATA_PATH = EXPERIMENT_PATH + "tensor_data"
    """ single_target(EXPERIMENT_PATH=EXPERIMENT_PATH,
                  DATA_PATH=DATA_PATH,
                  TENSOR_DATA_PATH=TENSOR_DATA_PATH,
                  window_sequence=temporal_sequences,
                  num_neurons=number_neurons, learning_rate=learning_rate,
                  testing_set=TEST_SET
                  )

    #visualization
    report_configurations(temporal_sequence=temporal_sequences,num_neurons=number_neurons,
                          experiment_folder=EXPERIMENT_PATH,results_folder="result",
                          report_folder="report",output_filename="overall_report")

    report_crypto(experiment_folder=EXPERIMENT_PATH,result_folder="result",report_folder="report",output_filename="report")


    generate_line_chart(EXPERIMENT_PATH,"ADA",temporal_sequences,number_neurons,["single_target"])"""


    #MULTITARGET
    #create horizontal dataset
    EXPERIMENT_PATH = "../modelling/techniques/forecasting/output/" + distance_measure + "/" + start_date + "_" + end_date + "/multi_target/"
    #datapath invariato
    #create_horizontal_dataset(DATA_PATH, EXPERIMENT_PATH)

    #VECTOR AUTOREGRESSION
    vector_autoregression(EXPERIMENT_PATH+"horizontal_dataset/horizontal.csv",TEST_SET)

    #MULTITARGET LSTM
main()