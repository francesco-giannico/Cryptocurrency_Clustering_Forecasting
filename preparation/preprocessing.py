from preparation.cleaning import remove_uncomplete_rows_by_range, input_missing_values, remove_outliers
from preparation.construction import min_max_scaling, max_abs_scaling, standardization
from preparation.integration import integrate_with_indicators
from preparation.selection import find_by_dead_before, find_uncomplete,remove_features
from utility.folder_creator import folder_creator

PATH_PREPROCESSED = "../preparation/preprocessed_dataset/"
PATH_CLEANED_FOLDER="../preparation/preprocessed_dataset/cleaned/final/"
PATH_MINMAXNORMALIZED_FOLDER="../preparation/preprocessed_dataset/constructed/min_max_normalized/"
PATH_MAXABSNORMALIZED_FOLDER="../preparation/preprocessed_dataset/constructed/max_abs_normalized/"
PATH_STANDARDIZED_FOLDER="../preparation/preprocessed_dataset/constructed/standardized/"
PATH_INTEGRATED_FOLDER="../preparation/preprocessed_dataset/integrated/"
def preprocessing():
    """folders_setup()
    feature_selection()
    separation()"""
    cleaning()
    integration()
    construction()


def folders_setup():
    # Set the name of folder in which to save all intermediate results
    folder_creator(PATH_PREPROCESSED,0)


def feature_selection():
    #remove_features(["Open","High","Adj Close","Low","Volume"])
    remove_features(["Adj Close","Volume"])

def separation():
    find_by_dead_before()
    find_uncomplete()

def cleaning():
    remove_outliers()
    """remove_uncomplete_rows_by_range("ARDR","2017-01-01","2019-12-31")
    remove_uncomplete_rows_by_range("REP", "2017-01-01", "2019-12-31")"""
    #todo LKK lo abbiamo rimosso perchè ha 144 missing values nel 2018!!
    #input_missing_values()



def integration():
    integrate_with_indicators(input_path=PATH_CLEANED_FOLDER)


def construction():
    #feature scaling
    min_max_scaling(input_path=PATH_INTEGRATED_FOLDER,output_path=PATH_MINMAXNORMALIZED_FOLDER)
    max_abs_scaling(input_path=PATH_INTEGRATED_FOLDER, output_path=PATH_MAXABSNORMALIZED_FOLDER)
    standardization(input_path=PATH_INTEGRATED_FOLDER, output_path=PATH_STANDARDIZED_FOLDER)

