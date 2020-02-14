from preprocessing.preprocessing import preprocessing

"""import os
import sys
import warnings
from crypto_clustering import consensus_clustering
from crypto_clustering.distanceMeasures.distance_computer import compute_distance_matrix
from crypto_runner.checknegatives import check_negatives
from crypto_runner.do_experiment_single import single_target
from crypto_utility.report_data import report_configurations, report_stockseries
from crypto_utility.writer import save_dummy_clusters

warnings.filterwarnings("ignore")
from math import sqrt
from crypto_runner.do_preprocessing import cut_crypto, create_horizontal_from_cut, preprocessing_main
from crypto_utility.dataset_filter import get_date_crypto_less_entry
from crypto_utility.reader import get_clusters, get_clusters2
from crypto_utility import test_set
from crypto_utility.folder_creator import folder_creator
from crypto_runner.multitarget_experiment import multi_target
from crypto_runner.reporting import generate_linechart_png, generate_averagermseForK,overall_report, results_by_cluster, barchartComparison, generate_fileaverageRMSE_byalgorithm

#todo:k_n sarebbero gli algoritmi, chiamalo algorithms
def experiment(k_n,folderoutput,folderpreprocessing):
    # clustering
    for c_algorithm in k_n:
        print("working on: "+c_algorithm)
        folder_creator("crypto_clustering/" + folderoutput + "/", 0)
        folder_creator("crypto_clustering/" + folderoutput + "/" + c_algorithm, 0)

        #TODO:modifica get_clusters2, non si capisce la differenza.
        clusters = get_clusters2(folderoutput,c_algorithm)

        cluster_id = 0
        for cluster in clusters:
            print("Cluster n: " + str(cluster_id))
            #set the day to use to cut the data (the first day to use in training, should be the first day of the cryptocurrency with less entry)
            #folderpreprocessing is step2_normalized
            first_day = get_date_crypto_less_entry(folderpreprocessing,cluster)
            #cut the dataset of this cluster
            #todo: perchè qui sta scritto: Noindexes? Non è un parametro?
            cut_crypto(first_day,cluster,cluster_id,c_algorithm,"NOindexes",folderoutput,folderpreprocessing)
            #generate the horizontal dataset starting from the cut datasets
            create_horizontal_from_cut(c_algorithm,cluster_id,"NOindexes",folderoutput)

             # -------- PARAMETERS ----------------------------------------------------------------------------
            # LSTM Parameters
            temporal_sequence_considered = [30, 100, 200]
            number_neurons_LSTM = [128, 256]
            #temporal_sequence_considered = [30]
            #number_neurons_LSTM = [128]
            learning_rate = 0.001  #forse va cambiato + sono i dataset e piu grande deve essere...

            # set dimension for last layer in multitarget
            dimension_last_layer=0

            for id in cluster:
                dimension_last_layer= len(cluster[id])
            print('dimensione ultimo layer: ' + str(dimension_last_layer))
            # Indicate the features that will be excluded from the scaling operations
            MULTI_features_to_exclude_from_scaling = []
            for i in range(dimension_last_layer):
                MULTI_features_to_exclude_from_scaling.append('Symbol_'+str(i+1))

            #
            # # -------- TENSORS ----------------------------------------------------------------------------
            # Tensor location

            # Generate tensors based on preprocessed data (Relaunch if data changes)

            #
            # # -------- TEST SET ----------------------------------------------------------------------------
            # Retrieve unused set
            TEST_SET = test_set.get_testset("crypto_testset/from_2016_07_01_until_2017_06_26/test_set.txt")
            #
            #
            # # ------------------------------------ EXPERIMENT ------------------------------------
            # folder_creator("crypto_clustering/experiments/" + c_algorithm+"/cluster_"+str(cluster_id),0)
            TENSOR_PATH = "crypto_clustering/"+ folderoutput+"/"+c_algorithm +"/cluster_"+str(cluster_id)+"/crypto_TensorData"
            EXPERIMENT = "crypto_clustering/"+ folderoutput+"/"+ c_algorithm+"/cluster_"+str(cluster_id)+"/MultiTarget_Data"
            DATA_PATH = "crypto_clustering/"+ folderoutput+"/horizontalDataset/"+ c_algorithm+"/cluster_"+str(cluster_id)+"/"
            multi_target(EXPERIMENT=EXPERIMENT, DATA_PATH=DATA_PATH, TENSOR_DATA_PATH=TENSOR_PATH,
                         temporal_sequence=temporal_sequence_considered,
                         number_neurons=number_neurons_LSTM, learning_rate=learning_rate,
                         dimension_last_layer=dimension_last_layer,
                         features_to_exclude_from_scaling=MULTI_features_to_exclude_from_scaling, testing_set=TEST_SET,type="indexes")

            multi_target(EXPERIMENT=EXPERIMENT, DATA_PATH=DATA_PATH, TENSOR_DATA_PATH=TENSOR_PATH,
                         temporal_sequence=temporal_sequence_considered,
                         number_neurons=number_neurons_LSTM, learning_rate=learning_rate,
                         dimension_last_layer=dimension_last_layer,
                         features_to_exclude_from_scaling=MULTI_features_to_exclude_from_scaling, testing_set=TEST_SET,
                         type="Noindexes")
            #check_negatives(cluster,EXPERIMENT)
            #generate_linechart_png(EXPERIMENT, "multitarget",temporal_sequence_considered,number_neurons_LSTM,cluster)
            #generate_final_report()
            cluster_id+=1

def experiment_single():
    # # ------------------------------------ EXPERIMENT SINGLE TARGET (single) ------------------------------------
    TEST_SET = test_set.get_testset("crypto_testset/from_2016_07_01_until_2017_06_26/test_set.txt")
    temporal_sequence_considered = [30, 100, 200]
    number_neurons_LSTM = [128, 256]
    #temporal_sequence_considered = [30]
    #number_neurons_LSTM = [128]
    learning_rate = 0.001  #forse va cambiato + sono i dataset e piu grande deve essere...
    SINGLE_features_to_exclude_from_scaling = ['Symbol']
    folderoutput="crypto_clustering/experiment_common_indicators/single_target"
    folder_creator("crypto_clustering/experiment_common_indicators",0)
    folder_creator(folderoutput, 0)
    TENSOR_PATH = "crypto_clustering/experiment_common_indicators/single_target/crypto_TensorData"
    EXPERIMENT = "crypto_clustering/experiment_common_indicators/single_target/"+ "SingleTarget_Data"
    DATA_PATH = "crypto_clustering/experiment_common_indicators/cutData/noclustering_indicators_singletarget/cluster_0/" #hanno già indici , eventualmente
    folderoutput="experiment_common_indicators"
    folderpreprocessing = "crypto_preprocessing_indicators"
    clusters = get_clusters2("experiment_common_indicators", "noclustering_indicators_multitarget")
    cluster_id = 0
    for cluster in clusters:
        # set the day to use to cut the data (the first day to use in training, should be the first day of the cryptocurrency with less entry)
        first_day = get_date_crypto_less_entry(folderpreprocessing, cluster)
        # "cut" dei dataset presenti in questo cluster
        cut_crypto(first_day, cluster, cluster_id, "noclustering_indicators_singletarget","indexes", folderoutput, folderpreprocessing)

    single_target(EXPERIMENT=EXPERIMENT, DATA_PATH=DATA_PATH, TENSOR_DATA_PATH=TENSOR_PATH,
                   temporal_sequence=temporal_sequence_considered,
                   number_neurons=number_neurons_LSTM, learning_rate=learning_rate,
                   features_to_exclude_from_scaling=SINGLE_features_to_exclude_from_scaling, testing_set=TEST_SET)

def results():
    k_n_consensus = ["consensus_k_sqrtN", "consensus_k_sqrtNdiv2", "consensus_k_sqrtNby2","consensus_k_sqrtNdiv4","consensus_k_sqrtNby4"]
    #k_n_consensus=["consensus_k_sqrtN"]
    experiments=['experiment_dtw_close','experiment_pearson_close',"experiment_dtw_allFeatures_noindicators","experiment_pearson_allFeatures_noindicators"]
    for experiment in experiments:
        for c_algorithm in k_n_consensus:
            clusters = get_clusters2(experiment,c_algorithm)
            generate_averagermseForK(path = "crypto_clustering/"+experiment+"/" +c_algorithm,num_of_clusters=len(clusters),name_experiment_model="MultiTarget_Data")

    #for single target
    generate_averagermseForK(path="crypto_clustering/experiment_common/noclustering_noindicators_singletarget",
                             num_of_clusters=1, name_experiment_model="SingleTarget_Data")
    experiments = ['experiment_dtw_close', 'experiment_pearson_close']
    #experiments = [ "experiment_dtw_allFeatures_noindicators","experiment_pearson_allFeatures_noindicators"]

    experiments.append('experiment_common')
    folder_creator("crypto_clustering/reports",0)
    filename="only_close_noindicators"
    generate_fileaverageRMSE_byalgorithm(path = "crypto_clustering/reports/",name_final_file=filename,experiments=experiments)
    overall_report("AVERAGE RMSE - NO INDICATORS - ONLY CLOSE",filename,filename)

    #results_by_cluster(path = "crypto_clustering/experiments/",algorithms=k_n_medoids)
    experiments = ['experiment_dtw_allFeatures_noindicators']
    barchartComparison(experiments)

def generate_clusters(k_type,distance_matrix,folderoutput):
    consensus_clustering.consensus_clustering(k_type,distance_matrix,folderoutput)
    pass

def generate_report():
    temporal_sequence = [30, 100, 200]
    number_neurons = [128, 256]
    # temporal_sequence_considered = [30]
    # number_neurons_LSTM = [128]
    learning_rate = 0.001
    folderoutput="experiment_common"


    #k_n_consensus = ["consensus_k_sqrtN", "consensus_k_sqrtNdiv2", "consensus_k_sqrtNby2", "consensus_k_sqrtNdiv4",
     #               "consensus_k_sqrtNby4"]
    k_n_consensus=['noclustering_noindicators_singletarget']
    for g in k_n_consensus:
        cluster_id = 0
        c_algorithm = g
        EXPERIMENT = "crypto_clustering/"+ folderoutput+"/"+ c_algorithm+"/"
        clusters = get_clusters2(folderoutput, c_algorithm)
        for cluster in clusters:
            EXPERIMENTFIN=EXPERIMENT + "cluster_"+str(cluster_id)+"/SingleTarget_Data"
            RESULT_PATH = "Result"
            REPORT_FOLDER_NAME = "Report"
            """report_configurations(temporal_sequence_used=temporal_sequence, neurons_used=number_neurons,
                                          name_folder_experiment=EXPERIMENTFIN, name_folder_result_experiment=RESULT_PATH,
                                          name_folder_report=REPORT_FOLDER_NAME, name_output_files="overall_report")

            report_stockseries(name_folder_experiment=EXPERIMENTFIN, name_folder_result_experiment=RESULT_PATH,
                                       name_folder_report=REPORT_FOLDER_NAME,
                                       name_files_output="report")"""
            generate_linechart_png(EXPERIMENTFIN,str(cluster_id),temporal_sequence,number_neurons,cluster)
            cluster_id+=1"""

def main():
    #todo preprocessing
    folderpreprocessing="../dataset/original"
    preprocessing("t")
    #todo calcolo la matrice delle distanze
    #todo dynamic time warping
    #compute_distance_matrix("dtw",[],"dtw_allFeatures_noindicators")
    #compute_distance_matrix("dtw", ['High','Low','Open'], "dtw_close_noindicators")
    #compute_distance_matrix("dtw", ['High','Low'], "dtw_openclose_noindicators")

    #compute_distance_matrix("dtw", ['High','Low','Open'], "dtw_close_indicators",folderpreprocessing=folderpreprocessing)
    #compute_distance_matrix("dtw", [], "dtw_allFeatures_indicators",folderpreprocessing=folderpreprocessing)
    #compute_distance_matrix("dtw", ['High','Low'], "dtw_closeopen_indicators")

    #todo pearson
    compute_distance_matrix("pearson", [], "pearson_allFeatures_noindicators")

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
    results()
    #compute_distance_matrix("euclidean", ['High','Low','Open'], "euclideantest",folderpreprocessing="crypto_preprocessing_noindicators")

    #generate_report()

main()