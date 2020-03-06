from modelling.techniques.clustering.ensemble_clustering.consensus_clustering import consensus_clustering
from modelling.techniques.clustering.distance_measures.distance_measures import compute_distance_matrix
from utility.clustering_utils import generate_cryptocurrencies_dictionary, prepare_dataset_for_clustering, \
    separate_folders
from utility.dataset_utils import cut_dataset_by_range
from utility.folder_creator import folder_creator
from utility.reader import get_dict_symbol_id


def clustering(distance_measure,start_date,end_date):
    CLUSTERING_PATH = "../modelling/techniques/clustering/output/" + distance_measure + "/" + start_date + "_" + end_date + "/"
    PATH_SOURCE = "../preparation/preprocessed_dataset/constructed/normalized/"
    #PATH_SOURCE = "../preparation/preprocessed_dataset/integrated/"

    folder_setup(CLUSTERING_PATH,distance_measure,start_date,end_date)

    prepare_dataset_for_clustering(start_date,end_date,input_path=PATH_SOURCE,output_path=CLUSTERING_PATH)
    
    generate_cryptocurrencies_dictionary(CLUSTERING_PATH+"cut_datasets/",CLUSTERING_PATH)
    
    dict_symbol_id = get_dict_symbol_id(CLUSTERING_PATH)
    
    compute_distance_matrix(dict_symbol_id,distance_measure,CLUSTERING_PATH)
    
    consensus_clustering(CLUSTERING_PATH)


def folder_setup(CLUSTERING_PATH,distance_measure,start_date,end_date):
    folder_creator(CLUSTERING_PATH, 1)
    folder_creator(CLUSTERING_PATH + "cut_datasets/", 1)
    return