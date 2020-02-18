import os
import shutil
import pandas as pd
from utility.cut import cut_dataset_by_range
from utility.folder_creator import folder_creator

PATH_PREPROCESSED_FOLDER="../preparation/preprocessed_dataset/"
PATH_UNCOMPLETE_FOLDER="../preparation/preprocessed_dataset/selected/uncomplete/"
PATH_COMPLETE_FOLDER="../preparation/preprocessed_dataset/selected/complete/"
PATH_CLEANED_FOLDER="../preparation/preprocessed_dataset/cleaned/"


def remove_uncomplete_rows_by_range(crypto_symbol,start_date,end_date):
 folder_creator(PATH_CLEANED_FOLDER,0)
 folder_creator(PATH_CLEANED_FOLDER+"partial", 0)
 df=cut_dataset_by_range(PATH_UNCOMPLETE_FOLDER,crypto_symbol,start_date,end_date)
 df.to_csv(PATH_CLEANED_FOLDER+"partial/"+crypto_symbol+".csv",sep=",",index=False)

def input_missing_values():
    folder_creator(PATH_CLEANED_FOLDER+"final",1)
    already_treated=[]
    for crypto_symbol in os.listdir(PATH_CLEANED_FOLDER+"partial"):
        df = pd.read_csv(PATH_CLEANED_FOLDER+"partial/"+crypto_symbol, delimiter=',', header=0)
        already_treated.append(crypto_symbol)
        df=interpolate_with_time(df)
        df.to_csv(PATH_CLEANED_FOLDER + "final/" + crypto_symbol, sep=",", index=False)

    for crypto_symbol in os.listdir(PATH_UNCOMPLETE_FOLDER):
        df = pd.read_csv(PATH_UNCOMPLETE_FOLDER+crypto_symbol, delimiter=',', header=0)
        if crypto_symbol not in already_treated:
            df=interpolate_with_time(df)
            df.to_csv(PATH_CLEANED_FOLDER + "final/" + crypto_symbol , sep=",", index=False)

    #merge with complete dataset
    for crypto_symbol in os.listdir(PATH_COMPLETE_FOLDER):
        shutil.copy(PATH_COMPLETE_FOLDER+ crypto_symbol, PATH_CLEANED_FOLDER+ "final/" + crypto_symbol)

#todo spiegare come mai hai scelto questo metodo di interpolazione... ce ne sono tanti a disposizione
def interpolate_with_time(df):
    # Converting the column to DateTime format
    df.Date = pd.to_datetime(df.Date, format='%Y-%m-%d')
    df = df.set_index('Date')
    # interpolate with time
    df = df.interpolate(method='time')
    df = df.reset_index()
    return df


