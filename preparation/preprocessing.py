from preparation.cleaning import remove_uncomplete_rows_by_range, input_missing_values, \
    remove_outliers_dbscan, remove_outliers_one
from preparation.construction import min_max_scaling, max_abs_scaling, standardization, robust_scaling
from preparation.integration import integrate_with_indicators, integrate_with_lag
from preparation.selection import find_by_dead_before, find_uncomplete,remove_features
from preparation.transformation import power_transformation, power_transformation2, quantile_transform, \
    quantile_transform2
from utility.folder_creator import folder_creator

PATH_PREPROCESSED = "../preparation/preprocessed_dataset/"
PATH_CLEANED_FOLDER= "../preparation/preprocessed_dataset/cleaned/final/"
PATH_MINMAXNORMALIZED_FOLDER= "../preparation/preprocessed_dataset/constructed/min_max_normalized/"
PATH_MAXABSNORMALIZED_FOLDER= "../preparation/preprocessed_dataset/constructed/max_abs_normalized/"
PATH_ROBUSTNORMALIZED_FOLDER= "../preparation/preprocessed_dataset/constructed/robust_normalized/"
PATH_STANDARDIZED_FOLDER= "../preparation/preprocessed_dataset/constructed/standardized/"
PATH_INTEGRATED_FOLDER= "../preparation/preprocessed_dataset/integrated/"
PATH_TRANSFORMED_FOLDER= "../preparation/preprocessed_dataset/transformed/"
PATH_NORMALIZED_FOLDER = "../preparation/preprocessed_dataset/constructed/"

def preprocessing():
    folders_setup()
    feature_selection()
    separation()
    cleaning()

    """quantile_transform(input_path=PATH_CLEANED_FOLDER,output_path=PATH_TRANSFORMED_FOLDER)
    integration(input_path=PATH_TRANSFORMED_FOLDER)
    quantile_transform2(input_path=PATH_INTEGRATED_FOLDER,output_path=PATH_TRANSFORMED_FOLDER)
    construction(input_path=PATH_TRANSFORMED_FOLDER)"""

    #transformation(input_path=PATH_TRANSFORMED_FOLDER,output_path=PATH_TRANSFORMED_INT_FOLDER)
    # transformation2(input_path=PATH_INTEGRATED_FOLDER,output_path=PATH_TRANSFORMED_FOLDER)
    integration(input_path=PATH_CLEANED_FOLDER)
    construction(input_path=PATH_INTEGRATED_FOLDER)
    #construction(input_path=PATH_CLEANED_FOLDER)


#todo aggiustare qua
def transformation2(input_path,output_path):
    power_transformation2(input_path,output_path)

def transformation(input_path,output_path):
    power_transformation(input_path,output_path)

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

def integration(input_path):
    integrate_with_indicators(input_path)
    #integrate_with_lag(input_path)

def construction(input_path):
    #feature scaling
    min_max_scaling(input_path,output_path=PATH_MINMAXNORMALIZED_FOLDER)
    #robust_scaling(input_path=PATH_TRANSFORMED_FOLDER,output_path=PATH_ROBUSTNORMALIZED_FOLDER)
    """max_abs_scaling(input_path,output_path=PATH_MAXABSNORMALIZED_FOLDER)
    standardization(input_path, output_path=PATH_STANDARDIZED_FOLDER)"""

