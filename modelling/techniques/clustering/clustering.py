from modelling.techniques.clustering.ensemble_clustering.consensus_clustering import consensus_clustering
from modelling.techniques.clustering.distance_measures.distance_measures import compute_distance_matrix
from utility.clustering_utils import generate_cryptocurrencies_dictionary, prepare_dataset_for_clustering
from utility.dataset_utils import cut_dataset_by_range
from utility.folder_creator import folder_creator
from utility.reader import get_dict_symbol_id

def clustering(distance_measure,start_date,end_date):
    CLUSTERING_PATH=folder_setup(distance_measure,start_date,end_date)
    #todo potare già i dataset che non vanno bene per questo range.
    #in output: wasserstain: 2018-2019 cut_dataset solo quelli validi.
    #crea un dic_symbol_id a partire da quelli che stanno nella folder cut.
    prepare_dataset_for_clustering(start_date,end_date,CLUSTERING_PATH)
    generate_cryptocurrencies_dictionary(CLUSTERING_PATH+"cut_datasets/",CLUSTERING_PATH)
    dict_symbol_id = get_dict_symbol_id(CLUSTERING_PATH)
    compute_distance_matrix(dict_symbol_id,distance_measure,start_date,end_date,CLUSTERING_PATH)
    consensus_clustering(CLUSTERING_PATH)

def folder_setup(distance_measure,start_date,end_date):
    PARTIAL_PATH = \
        "../modelling/techniques/clustering/output/"+ distance_measure + "/" + start_date + "_" + end_date + "/"
    folder_creator(PARTIAL_PATH, 1)
    folder_creator(PARTIAL_PATH + "cut_datasets/", 1)
    return PARTIAL_PATH