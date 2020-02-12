import shutil

import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from pandas.plotting import lag_plot
def test():
   #print(df.columns.values.tolist())
   columns=['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
   withNull = []
   for file in os.listdir("../dataset/original/"):
    df = pd.read_csv("../dataset/original/"+file, delimiter=',',header=0)
    df=df.set_index("Date")
    #print(df.index)
    #index on date
    #print(df.head(7))
    #print(df.describe())
    #missing values
    i=0

    for column in columns:
      if(df[column].isnull().any()):
         #print(file)
         shutil.move("../dataset/original/"+file, "../dataset/with_null_values/"+file)
         withNull.append(file)
         break
   print(len(withNull))


   """
    feature = "Adj Close"
    plt.figure(num=None, figsize=(20, 6))
    plt.subplot(1, 2, 1)
    ax = df[feature].plot(style=['-'])
    ax.lines[0].set_alpha(0.3)
    ax.set_ylim(0, np.max(df[feature]+0.3))
    plt.xticks(rotation=90)
    plt.title("No scaling")
    ax.legend()

    plt.subplot(1, 2, 2)
    ax = df[feature].plot(style=['-'])
    ax.lines[0].set_alpha(0.3)
    ax.set_yscale('log')
    #ax.set_ylim(0, np.max(df['Close']+1))
    plt.xticks(rotation=90)
    plt.title("logarithmic scale")
    ax.legend()
    #plt.show()

    #lag_plot(df[feature])
    plt.show()"""

test()