import os
from datetime import datetime
from decimal import Decimal
from itertools import product
from math import sqrt
import pandas as pd
from modelling.techniques.baseline.simple_prediction.simple_prediction import simple_prediction
from modelling.techniques.baseline.vector_autoregression.vector_autoregression import vector_autoregression
from modelling.techniques.clustering.clustering import clustering
from modelling.techniques.forecasting.multi_target import multi_target
from modelling.techniques.forecasting.single_target import single_target
from modelling.techniques.forecasting.testing.test_set import generate_testset, get_testset, generate_testset2
from preparation.construction import create_horizontal_dataset
from preparation.preprocessing import preprocessing
from acquisition.yahoo_finance_history import get_most_important_cryptos
from understanding.exploration import describe
from utility.clustering_utils import merge_predictions
from utility.dataset_utils import cut_dataset_by_range
from utility.folder_creator import folder_creator
from visualization.bar_chart.forecasting import report_configurations, report_crypto
from visualization.line_chart import generate_line_chart
import numpy as np



def main():
    #DATA UNDERSTANDING
    cryptocurrencies=['BTC','ETH']
    #data_understanding(cryptocurrencies)

    #DATA PREPARATION
    preprocessing()
    #Description after
    types="min_max_normalized"
    #features_to_use=['Close','Open','Low','High','RSI_14','RSI_30','RSI_60','SMA_30','SMA_60','SMA_14','EMA_14','EMA_30','EMA_60']
    """features_to_use = ['Close', 'Open', 'Low', 'High', 'RSI_14', 'RSI_7', 'RSI_20', 'SMA_7', 'SMA_14', 'SMA_20',
                       'EMA_7', 'EMA_14', 'EMA_20']
    features_to_use = ['Close', 'Open', 'Low', 'High', 'RSI_30', 'RSI_60', 'RSI_100', 'SMA_30', 'SMA_60', 'SMA_100',
                       'EMA_30', 'EMA_60', 'EMA_100']"""
    features_to_use = ['Close', 'Open', 'Low', 'High','Adj Close']
    #features_to_use=[]
    """describe(PATH_DATASET="../preparation/preprocessed_dataset/constructed/"+types+"/",
             output_path="../preparation/preprocessed_dataset/",
             name_folder_res=types,
             features_to_use=features_to_use)"""

    #CLUSTERING
    """start_date = "2014-10-01"
    end_date = "2019-12-31"""
    distance_measure = "wasserstain"
    #clustering_main(distance_measure,start_date,end_date)

    #TESTING SET
    TEST_SET=testing_set()

    #MODELLING
    #features_to_use = ['Date','Close','Open','High','Low','Adj Close']
    #features_to_use = ['Date', 'Close', 'Open', 'High', 'Low', 'Adj Close','RSI_14','RSI_30','RSI_60']
    #['Date', 'Close']
    """list_features_to_uses=[
            ['Date','Close','Open','Close','Low','High','Adj Close'],
    ]"""
    list_features_to_uses = [
        ['Date', 'Close', 'Open', 'High', 'Low', 'Adj Close', 'RSI_14', 'RSI_30', 'RSI_60'],
        ['Date', 'Close', 'Open', 'High', 'Low', 'Adj Close'],
    ]
    #todo this one
    temporal_sequences = [[45],[75],[90],[100],[115],[130],[220]]
    number_neurons = [[128],[256]]

    #DONE
    """temporal_sequences = [[15],[30],[50],[60],[120],[150]]
    number_neurons = [[128],[256]]"""

    """temporal_sequences = [[30],[50],[60],[120],[150]]
    number_neurons = [[256]]"""

    """temporal_sequences = [[30]]
    number_neurons = [[128]]"""


    learning_rate = 0.001
    # General parameters
    DROPOUT = 0.45
    EPOCHS = 100
    BATCH_SIZE = 2000
    #todo rimuovi quando farai clustering
    start_date = "2010-01-01"
    end_date = "2019-12-31"
    for features_to_use in list_features_to_uses:
        print("Features: "+ str(features_to_use))
        for num in number_neurons:
            for temp_seq in temporal_sequences:
                out = ""
                for ft in features_to_use:
                    out += ft + "_"
                output_name = out + "neur{}-dp{}-ep{}-bs{}-lr{}-tempseq{}.csv".format(num[0], DROPOUT, EPOCHS,
                                                                                      BATCH_SIZE, learning_rate,
                                                                                      temp_seq[0])
                single_target_main(distance_measure,start_date,end_date,TEST_SET,types,features_to_use,temp_seq,num,learning_rate,DROPOUT,EPOCHS,BATCH_SIZE,output_name)

    #report()
    #MULTITARGET
    #multi_target_main(distance_measure,start_date,end_date,TEST_SET)

def single_target_main(distance_measure,start_date,end_date,TEST_SET,type,features_to_use,temporal_sequences,number_neurons,learning_rate,DROPOUT,EPOCHS,BATCH_SIZE,output_name):
    DATA_PATH = "../preparation/preprocessed_dataset/constructed/"+type+"/"
    #DATA_PATH="../modelling/techniques/clustering/output/" + distance_measure + "/" + start_date + "_" + end_date

    # SIMPLE PREDICTION
    #DATA_PATH_SIMPLE="../acquisition/dataset/original/"
    DATA_PATH_SIMPLE = DATA_PATH
    OUTPUT_SIMPLE_PREDICTION= "../modelling/techniques/baseline/simple_prediction/output/"
    simple_prediction(DATA_PATH_SIMPLE,TEST_SET,OUTPUT_SIMPLE_PREDICTION)

    # SINGLE TARGET LSTM
    """EXPERIMENT_PATH = "../modelling/techniques/forecasting/output/" + distance_measure + "/" + start_date + "_" + end_date + "/single_target/"
    TENSOR_DATA_PATH = EXPERIMENT_PATH + "tensor_data"""
    EXPERIMENT_PATH = "../modelling/techniques/forecasting/outputs/"+output_name+"/single_target/"
    TENSOR_DATA_PATH = EXPERIMENT_PATH + "tensor_data"

    single_target(EXPERIMENT_PATH=EXPERIMENT_PATH,
                  DATA_PATH=DATA_PATH,
                  TENSOR_DATA_PATH=TENSOR_DATA_PATH,
                  window_sequence=temporal_sequences,
                  list_num_neurons=number_neurons, learning_rate=learning_rate,
                  testing_set=TEST_SET,features_to_use=features_to_use,
                  DROPOUT=DROPOUT,EPOCHS=EPOCHS,BATCH_SIZE=BATCH_SIZE
                  )

    # visualization single_target
    """report_configurations(temporal_sequence=temporal_sequences,num_neurons=number_neurons,
                          experiment_folder=EXPERIMENT_PATH,results_folder="result",
                          report_folder="report",output_filename="overall_report")"""

    #report_crypto(experiment_folder=EXPERIMENT_PATH,result_folder="result",report_folder="report",output_filename="report")

    #merge_predictions(EXPERIMENT_PATH,"result")


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
                    vector_autoregression(EXPERIMENT_PATH + "clusters" + "/" + k_used+"/"+cluster+"/"+"horizontal_dataset/horizontal.csv",TEST_SET, OUTPUT_FOLDER_VAR,cryptos_in_the_cluster)

                    # LSTM
                    dim_last_layer = len(cryptos_in_the_cluster)
                    features_to_use=['Date']
                    for i in range(len(cryptos_in_the_cluster)):
                        features_to_use.append("Close_"+str(i+1))
                    """multi_target(EXPERIMENT_PATH=EXPERIMENT_PATH + "clusters" + "/" + k_used+"/"+cluster+"/",
                                      DATA_PATH=EXPERIMENT_PATH + "clusters" + "/" + k_used+"/"+cluster+"/"+"horizontal_dataset/",
                                      TENSOR_DATA_PATH=EXPERIMENT_PATH + "clusters" + "/" + k_used+"/"+cluster+"/" "tensor_data/",
                                      window_sequence=temporal_sequences,
                                      list_num_neurons=number_neurons, learning_rate=learning_rate,
                                      dimension_last_layer=dim_last_layer,testing_set=TEST_SET,cryptos=cryptos_in_the_cluster,features_to_use=features_to_use)
"""
                    """ report_configurations(exp_type="multi_target",temporal_sequence=temporal_sequences,num_neurons=number_neurons,
                                            experiment_folder=EXPERIMENT_PATH + "clusters" + "/" + k_used+"/"+cluster+"/",results_folder="result",
                                            report_folder="report",output_filename="overall_report")
                    """
                    # report_crypto(experiment_folder=EXPERIMENT_PATH + "clusters" + "/" + k_used+"/"+cluster+"/",result_folder="result",report_folder="report",output_filename="report")

                    # generate_line_chart(EXPERIMENT_PATH + "clusters" + "/" + k_used+"/"+cluster+"/",temporal_sequences,number_neurons)

                    # other charts for clustering

def report():
    OUTPUT_SIMPLE_PREDICTION = "../modelling/techniques/baseline/simple_prediction/output/"

    file = open(OUTPUT_SIMPLE_PREDICTION + "average_rmse/average_rmse.txt", "r")
    value1 = file.read()

    # baseline wins on the following cryptocurrencies:
    folder_creator("../modelling/techniques/comparisons/single_target/", 0)
    filename = '../modelling/techniques/comparisons/single_target/'
    gen_path = "../modelling/techniques/forecasting/outputs/"
    for folder in os.listdir(gen_path):
        df_out = {"symbol": [], "baseline": [], "single_target": [], "is_best": [], "distance_from_bs": []}
        rmses = []
        for subfold in os.listdir(gen_path + folder + "/single_target/result/"):
            for subfold2 in os.listdir(gen_path + folder + "/single_target/result/" + subfold + "/"):
                df1 = pd.read_csv(
                    gen_path + folder + "/single_target/result/" + "/" + subfold + "/" + subfold2 + "/stats/errors.csv",
                    usecols=['rmse_norm'])
                file = open(OUTPUT_SIMPLE_PREDICTION + "average_rmse/" + subfold, "r")
                value_bas = float(file.read())
                df_out['symbol'].append(subfold)
                df_out['baseline'].append(value_bas)
                df_out['single_target'].append(df1['rmse_norm'][0])
                is_best = False
                if df1['rmse_norm'][0] < value_bas:
                    is_best = True
                df_out['is_best'].append(is_best)
                distance = np.abs(df1['rmse_norm'][0] - value_bas)
                df_out['distance_from_bs'].append(distance)
                rmses.append(df1['rmse_norm'][0])

            pd.DataFrame(data=df_out).to_csv(filename + folder, index=False)
            value = np.mean(rmses)
            print("Baseline (AVG RMSE): " + str(value1))
            print("Single Target (AVG RMSE): " + str(value))

    path = "../modelling/techniques/comparisons/single_target/"
    path_out = "../modelling/techniques/comparisons/"
    df_out = pd.DataFrame()
    """min=1000
    min_name="""""
    for experiment_name in os.listdir(path):
        df = pd.read_csv(path + experiment_name)
        df_out['symbol'] = df['symbol']
        df_out['baseline'] = df['baseline']
        df_out[experiment_name] = df['single_target']

    df_out.to_csv(path_out + "final.csv", index=False)
def data_understanding(crypto_names):
    #DATA UNDERSTANDING
    PATH_DATASET= "../acquisition/dataset/original/"

    #COLLECT INITIAL DATA
    #data collecting from yahoo finance
    #get_most_important_cryptos(crypto_names,startdate=datetime(2010, 1, 2),enddate=datetime(2020, 1, 1))

    # EXPLORE DATA
    #missing_values(PATH_DATASET)
    #describe dataframes
    OUTPUT_PATH="../understanding/output/"
    describe(PATH_DATASET,OUTPUT_PATH,None,None)

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




main()