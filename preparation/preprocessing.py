from preparation.cleaning import remove_uncomplete_rows_by_range, input_missing_values
from preparation.construction import min_max_scaling
from preparation.integration import integrate_with_indicators
from preparation.selection import find_by_dead_before, find_uncomplete,remove_features
from utility.folder_creator import folder_creator

PATH_PREPROCESSED = "../preparation/preprocessed_dataset/"

def preprocessing():
    folders_setup()
    feature_selection()
    separation()
    cleaning()
    construction()
    integration()

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
    remove_uncomplete_rows_by_range("ARDR","2017-01-01","2019-12-31")
    remove_uncomplete_rows_by_range("REP", "2017-01-01", "2019-12-31")
    #todo LKK lo abbiamo rimosso perchè ha 144 missing values nel 2018!!
    input_missing_values()

def construction():
    #feature scaling
    min_max_scaling()

def integration():
    integrate_with_indicators()