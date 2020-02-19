import os

from utility.folder_creator import folder_creator
from utility.reader import get_preprocessed_crypto_symbols
from utility.writer import save_dict_symbol_id
import pandas as pd

PATH_OUTPUT="../modelling/techniques/clustering/output/"
def generate_cryptocurrencies_dictionary():
    crypto_symbols = get_preprocessed_crypto_symbols()
    df = pd.DataFrame(columns=['id'],index=crypto_symbols)
    i=0
    for crypto_name in crypto_symbols:
        df.at[crypto_name, "id"] = i
        i+=1
    df = df.rename_axis('symbol').reset_index()
    df.to_csv(PATH_OUTPUT+"symbol_id.csv",",",index=False)
