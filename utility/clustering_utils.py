import os
from utility.reader import get_preprocessed_crypto_symbols
from utility.writer import save_dict_symbol_id


def generate_cryptocurrencies_dictionary():
    crypto_symbols = get_preprocessed_crypto_symbols()
    length = len(crypto_symbols)
    rg = range(length)
    dictionary = dict(zip(rg, crypto_symbols))
    save_dict_symbol_id(dictionary)
    return length,dictionary