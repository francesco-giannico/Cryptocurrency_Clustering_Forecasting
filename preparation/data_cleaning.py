# Import the required libraries
import math
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#todo completare, cioè cambiare algoritmo: quando trova una nulla, allora controlla
from utility.folder_creator import folder_creator

PATH_PREPROCESSED_FOLDER="../preparation/preprocessed_dataset/"
PATH_UNCOMPLETE_FOLDER="../preparation/preprocessed_dataset/selected/uncomplete/"
def remove_uncomplete_rows_by_range(crypto_symbol,start_date,end_date):
 folder_creator(PATH_UNCOMPLETE_FOLDER+"cleaned",1)
 folder_creator(PATH_UNCOMPLETE_FOLDER + "no_missing_values_part1", 1)
 df = pd.read_csv(PATH_UNCOMPLETE_FOLDER + crypto_symbol+".csv", delimiter=',', header=0)
 df=df.set_index("Date")
 df1 = df[(df.index <= start_date) | (df.index >= end_date)]
 df1 = df1.reset_index()
 df1.to_csv(PATH_PREPROCESSED_FOLDER+"cleane")




def input_missing_values():
    # To load the raw data:
    df = pd.read_csv("../acquisition/dataset/with_null_values/ARDR.csv", delimiter=',', header=0)
    # Converting the column to DateTime format
    df.Date = pd.to_datetime(df.Date, format='%Y-%m-%d')
    df = df.set_index('Date')
    # interpolate with time
    df = df.interpolate(method='time')
    df.to_csv("../dataset/interpolated/time/ARDR.csv", ",")
