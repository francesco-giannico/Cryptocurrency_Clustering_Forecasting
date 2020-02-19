from modelling.techniques.clustering.ensemble_clustering.consensus_clustering import consensus_clustering
from modelling.techniques.clustering.distance_measures.distance_measures import compute_distance_matrix
from utility.clustering_utils import generate_cryptocurrencies_dictionary
from utility.cut import cut_dataset_by_range
from utility.folder_creator import folder_creator
from utility.reader import get_dict_symbol_id

def clustering(distance_measure,start_date,end_date):
    CLUSTERING_PATH=folder_setup(distance_measure,start_date,end_date)

    generate_cryptocurrencies_dictionary()
    dict_symbol_id = get_dict_symbol_id()
    #todo potare già i dataset che non vanno bene per questo range.
    #in output: wasserstain: 2018-2019 cut_dataset solo quelli validi.
    #crea un dic_symbol_id a partire da quelli che stanno nella folder cut.
    cut_datasets(start_date,end_date)
    dict_length = dict_symbol_id.symbol.count()
    PATH_SOURCE = "../preparation/preprocessed_dataset/integrated/"
    for i in range(dict_length):
        try:
         df = cut_dataset_by_range(PATH_SOURCE, dict_symbol_id.symbol[i], start_date, end_date)
         df=df.set_index("Date")
         if(df.index[0]==start_date):
             df=df.reset_index()
             df.to_csv(CLUSTERING_PATH + "cut_dataset/" + dict_symbol_id.symbol[i] + ".csv", sep=",",index=False)
        except:
           pass

    """generate_cryptocurrencies_dictionary()
    dict_symbol_id = get_dict_symbol_id()"""
    #compute_distance_matrix(dict_symbol_id,distance_measure,start_date,end_date,CLUSTERING_PATH)
    #consensus_clustering(distance_measure,CLUSTERING_PATH)

def folder_setup(distance_measure,start_date,end_date):
    PARTIAL_PATH = "../modelling/techniques/clustering/output/"
    INT_PATH = PARTIAL_PATH + distance_measure + "/"
    FINAL_PATH = PARTIAL_PATH + distance_measure + "/" + start_date + "_" + end_date + "/"
    folder_creator(PARTIAL_PATH, 0)
    folder_creator(INT_PATH, 0)
    folder_creator(FINAL_PATH, 1)
    folder_creator(FINAL_PATH + "cut_dataset/", 1)
    return FINAL_PATH