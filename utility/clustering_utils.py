import os

from utility.cut import cut_dataset_by_range
from utility.folder_creator import folder_creator
from utility.reader import get_preprocessed_crypto_symbols, get_dict_symbol_id
from utility.writer import save_dict_symbol_id
import pandas as pd


PATH_SOURCE = "../preparation/preprocessed_dataset/integrated/"

def generate_cryptocurrencies_dictionary(PATH_TO_READ,PATH_OUTPUT):
    crypto_symbols = get_preprocessed_crypto_symbols(PATH_TO_READ)
    df = pd.DataFrame(columns=['id'],index=crypto_symbols)
    i=0
    for crypto_name in crypto_symbols:
        df.at[crypto_name, "id"] = i
        i+=1
    df = df.rename_axis('symbol').reset_index()
    df.to_csv(PATH_OUTPUT+"symbol_id.csv",",",index=False)

#selects only the datasets which cover the period of time of interest.
def prepare_dataset_for_clustering(start_date,end_date,CLUSTERING_PATH):
    for crypto in os.listdir(PATH_SOURCE):
        try:
            df = cut_dataset_by_range(PATH_SOURCE, crypto.replace(".csv",""), start_date, end_date)
            df = df.set_index("Date")
            if (df.index[0] == start_date):
                df = df.reset_index()
                df.to_csv(CLUSTERING_PATH + "cut_datasets/" + crypto, sep=",", index=False)
        except:
            pass
