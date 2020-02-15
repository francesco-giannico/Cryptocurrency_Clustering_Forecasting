# Import the required libraries
import math
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Import a scoring metric to compare methods
from sklearn.metrics import r2_score

#todo completare, cioè cambiare algoritmo: quando trova una nulla, allora controlla
#per anno: quanti sono quelli mancanti nel 2016?
#quanti sono quelli mancanti nel 2017? ecc


#se la percentuale di missing values in quel mese è alta, allora elimina il mese + i mesi
#precedenti

def remove_uncomplete_rows():
  for file in os.listdir("../data_acquisition/dataset/selected/uncomplete"):
    df = pd.read_csv("../data_acquisition/dataset/selected/uncomplete/" + "ARDR.csv", delimiter=',', header=0)
    df = df.set_index("Date")
    total_null=df.isnull().sum()[1]
    print("total null rows: "+ str(total_null))
    if(total_null>15):
        init_date = df.index[0]
        #get year
        first_year=int(init_date.split("-")[0])
        actual_year=first_year
        while actual_year < 2020:
            #next year
            actual_year+=1
            init_date=str(first_year-1)+"-12-31"
            fin_date=str(actual_year)+"-01-01"
            df1 = df.query('index<@fin_date')
            df1=df1.query('index >@init_date')
            #number of null of this year
            null_year=df1.isnull().sum()[1]
            #print(null_year)
            percentage_of_null=(null_year/total_null)*100
            #if(percentage_of_null>50):
            if (percentage_of_null == 0):
                print("In " + str(first_year) + " no missing values.")
                first_year = actual_year
            else:
                print("In "+str(first_year)+" there is the "+str(percentage_of_null)+ "% of null values")
                first_year=actual_year
                #remove the year
                """init_date=str(first_year)+"-12-31"
                print(init_date)
                df = df.query('index > @init_date')
                print(df)"""

            """ for row in df.itertuples():
                    #print(row.Open)
                    if (math.isnan(row.Open)):
                        fin_date=row.Index
                        #df=df.drop(df.index[init_date:row.Index])
                        df=df.query('index < @init_date or index > @fin_date')
                        init_date=row.Index
            df.to_csv('../dataset/reviewed/'+file,",")   """
    else:
        print("too low frequency")



def input_missing_values():
    # To load the raw data:
    df = pd.read_csv("../data_acquisition/dataset/with_null_values/ARDR.csv", delimiter=',', header=0)
    # Converting the column to DateTime format
    df.Date = pd.to_datetime(df.Date, format='%Y-%m-%d')
    df = df.set_index('Date')
    # interpolate with time
    df = df.interpolate(method='time')
    df.to_csv("../dataset/interpolated/time/ARDR.csv", ",")
