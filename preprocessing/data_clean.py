# Import the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Import a scoring metric to compare methods
from sklearn.metrics import r2_score

# To load the raw data:

df = pd.read_csv("../dataset/with_null_values/ARDR.csv", delimiter=',', header=0)
# Converting the column to DateTime format
df.Date = pd.to_datetime(df.Date, format='%Y-%m-%d')
df = df.set_index('Date')

#2018-01-14,1.93852,2.472475,1.71167,2.07228,2.07228,5524735.0
#2018-01-14,1.93852,2.472475,1.71167,2.07228,2.07228,5524735.0
#2018-01-14,1.93852,2.472475,1.71167,2.07228,2.07228,5524735.0
#2018-01-14,1.9799403100080113,2.4855895505011176,1.7536953896544811,2.2152743039771017,2.2152743039771017,5985795.030959155
#interpolate with time
df=df.interpolate(method='time')
"""df.Close=df.Close.interpolate(method='time')
df.High=df.High.interpolate(method='time')
df.Low=df.Low.interpolate(method='time')
df.AdjClose=df.AdjClose.interpolate(method="time")
df.Volume=df.Volume.interpolate(method="time")"""
df.to_csv("../dataset/interpolated/time/ARDR.csv",",")

#interpolate with

