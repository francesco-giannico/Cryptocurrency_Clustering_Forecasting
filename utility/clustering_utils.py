import os
from utility.reader import get_preprocessed_crypto_symbols
from utility.writer import save_dict_symbol_id
import pandas as pd

def generate_cryptocurrencies_dictionary():
    crypto_symbols = get_preprocessed_crypto_symbols()
    df = pd.DataFrame(columns=['id'],index=crypto_symbols)
    i=0
    for crypto_name in crypto_symbols:
        df.at[crypto_name, "id"] = i
        i+=1
    df = df.rename_axis('symbol').reset_index()
    df.to_csv("../modelling/techniques/clustering/symbol_id.csv",",",index=False)
    """df=pd.read_csv("../modelling/techniques/clustering/symbol_id.csv",sep=",",header=0,index_col=1)
    print(df.symbol[103])
    """