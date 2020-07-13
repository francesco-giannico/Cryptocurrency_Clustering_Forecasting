import os
from preparation.cleaning import remove_outliers_one
from preparation.construction import min_max_scaling, max_abs_scaling, add_trend_feature
from preparation.integration import integrate_with_indicators
from preparation.selection import find_by_dead_before, find_uncomplete,remove_features
from utility.dataset_utils import cut_dataset_by_range
from utility.folder_creator import folder_creator

PATH_PREPROCESSED = "../preparation/preprocessed_dataset/"
PATH_CLEANED_FOLDER= "../preparation/preprocessed_dataset/cleaned/final/"
PATH_MINMAXNORMALIZED_FOLDER= "../preparation/preprocessed_dataset/constructed/min_max_normalized/"
PATH_MINMAXMEANNORMALIZED_FOLDER= "../preparation/preprocessed_dataset/constructed/min_max_mean_normalized/"
PATH_MAXABSNORMALIZED_FOLDER= "../preparation/preprocessed_dataset/constructed/max_abs_normalized/"
PATH_ROBUSTNORMALIZED_FOLDER= "../preparation/preprocessed_dataset/constructed/robust_normalized/"
PATH_STANDARDIZED_FOLDER= "../preparation/preprocessed_dataset/constructed/standardized/"
PATH_INTEGRATED_FOLDER= "../preparation/preprocessed_dataset/integrated/"
PATH_TRANSFORMED_FOLDER= "../preparation/preprocessed_dataset/transformed/"
PATH_NORMALIZED_FOLDER = "../preparation/preprocessed_dataset/constructed/"
PATH_CUT_FOR_CLUSTERING = "../preparation/preprocessed_dataset/cut_for_clustering/"

def preprocessing(TEST_SET,start_date,percentual,end_date_for_clustering=None):
    """folders_setup()
    feature_selection()
    separation()
    cleaning()
    cut_datasets_for_clustering(input_path=PATH_CLEANED_FOLDER, output_path=PATH_CUT_FOR_CLUSTERING,
                                start_date=start_date, end_date_for_clustering=end_date_for_clustering)"""

    integration(input_path=PATH_CLEANED_FOLDER, output_path=PATH_INTEGRATED_FOLDER,test_set=TEST_SET, start_date=start_date,percent=percentual)
    construction(input_path=PATH_INTEGRATED_FOLDER,output_path=PATH_MAXABSNORMALIZED_FOLDER,type="max_abs")
    #construction(input_path=PATH_CUT_FOR_CLUSTERING,output_path=PATH_MINMAXNORMALIZED_FOLDER,type="min_max")

def cut_datasets_for_clustering(input_path,output_path,start_date,end_date_for_clustering):
    folder_creator(output_path,1)
    for crypto in os.listdir(input_path):
        df=cut_dataset_by_range(input_path,crypto.replace(".csv",""),start_date,end_date_for_clustering)
        df.to_csv(output_path+crypto,index=False)

def folders_setup():
    # Set the name of folder in which to save all intermediate results
    folder_creator(PATH_PREPROCESSED,0)

def feature_selection():
    #remove_features(["Open","High","Adj Close","Low","Volume"])
    remove_features([])
    #pass

def separation():
    find_by_dead_before()
    find_uncomplete()

def cleaning():
    #remove_outliers_dbscan()
    remove_outliers_one()
    """remove_uncomplete_rows_by_range("ARDR","2017-01-01","2019-12-31")
    remove_uncomplete_rows_by_range("REP", "2017-01-01", "2019-12-31")"""
    #todo LKK lo abbiamo rimosso perchè ha 144 missing values nel 2018!!
    #input_missing_values()

def integration(input_path,output_path,start_date,test_set,percent):
    #integrate_with_indicators(input_path,output_path,start_date,test_set)
    # add qualitative feature Trend
    add_trend_feature(input_path=output_path, output_path=output_path,percent=percent)
    #integrate_with_lag(input_path)

def construction(input_path,output_path,type):
    #feature scaling
    if type=="min_max":
        min_max_scaling(input_path,output_path)
    elif type=="max_abs":
        max_abs_scaling(input_path,output_path)
    else:
        print("Error in scaling!")
