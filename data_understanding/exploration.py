import itertools
import shutil
import pandas as pd
import os
import math
import matplotlib.pyplot as plt
import numpy as np
from pandas.plotting import lag_plot
from scipy.stats import pearsonr, stats

COLUMNS=['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
#it separates files starting from the original dataset, the ones with null values are moved in with_null_values folder and the ones
#dead before 31-12-2019 are moved in dead_before folder
def separate_files():
   #print(df.columns.values.tolist())
   for file in os.listdir("../data_acquisition/dataset/original/"):
    df = pd.read_csv("../dataset/original/"+file, delimiter=',',header=0)
    df=df.set_index("Date")

    #dead before
    lastDate=df.index[::-1][0]
    if lastDate!='2019-12-31':
        shutil.move("../dataset/original/" + file, "../dataset/dead_before/" + file)

    #with null values
    for column in COLUMNS:
      if(df[column].isnull().any()):
         #print(file)
         try:
            shutil.move("../dataset/original/"+file, "../dataset/with_null_values/"+file)
         except:
             pass
         break


#generates a file in which, for each cryptocurrency, there is the count of the missing values by column
def missing_values():
   crypto=[]
   for file in os.listdir("../data_acquisition/dataset/with_null_values/"):
       crypto.append(file.replace(".csv",""))
   df = pd.DataFrame(columns=COLUMNS, index=crypto)
   for file in os.listdir("../data_acquisition/dataset/with_null_values/"):
    df1 = pd.read_csv("../dataset/with_null_values/"+file, delimiter=',',header=0)
    df1=df1.set_index("Date")
    #counting the number of null element, for column
    # sf=
    cryptocurrency=file.replace(".csv","")
    df.loc[cryptocurrency]=df1.isnull().sum()
    #from series to dataframe
    #df1=pd.DataFrame({'column': sf.index, 'count': sf.values})
    df.to_csv("missing_values.csv",",")

def describe(df):
    print(df.describe())
    pass

def log_scaling(df):
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


    plt.show()

def lag_plot_mine(df):
    feature = "Adj Close"
    lag_plot(df[feature])
    plt.show()

def info_bivariate(data, features_name):
    thre = 0.4
    d = np.array(data)
    data_t = np.transpose(d)
    el = np.arange(0, len(data_t))
    combo_index = list(itertools.product(el, repeat=2))
    fig3 = plt.figure(figsize=[300, 500], dpi=100, facecolor='w', edgecolor='black')
    i = 1
    for e in combo_index:
        ind1 = e.__getitem__(0)
        ind2 = e.__getitem__(1)
        c, t = pearsonr(data_t[ind1], data_t[ind2])
        titolo = '\n{} - {}\nP: --> {}'.format(features_name[ind1], features_name[ind2], round(c, 2))
        #print(titolo)
        if c < thre and c > -thre:
             plot_correlationbtw2V(titolo, data_t[ind1], data_t[ind2], len(data_t), len(data_t), i, 'r*')
        else:
             plot_correlationbtw2V(titolo, data_t[ind1], data_t[ind2], len(data_t), len(data_t), i, 'g.')
        i = i + 1
    fig3.show()
    plt.show()
    return

def plot_correlationbtw2V(title, data1, data2, righe, colonne, indice, cm):
    plt.subplot(righe, colonne, indice)
    plt.plot(data1, data2, cm)
    # plt.tight_layout()
    plt.subplots_adjust(left=-0.2, right=0.8, top=0.8, bottom=-0.5)
    plt.title(title)
    return

def plot_boxnotch_univariateanalysis(data, features_name):
    fig2 = plt.figure(2, figsize=[10, 10], dpi=95, facecolor='w', edgecolor='black')
    numero_features = len(data)

    d = []
    for f in range(0, numero_features, 1):
        d.append(list(data[f]))
    plt.boxplot(d, notch=True)
    plt.title(f)

    fig2.show()
    plt.show()
    return

def info_univariate(data, features_name):
    d = np.array(data)
    data_t = np.transpose(d)
    for f in range(0, len(data_t), 1):
        ds = sorted(data_t[f])
        moda = stats.mode(ds)
        print('Feature: {}:\nMAX: --> {}\nMIN:  --> {}\nAVG:  --> {}\nMODE:  --> V:{} --> {}\nMed  --> {}\n'.format(
             features_name[f], np.max(data_t[f]),
             np.min(data_t[f]),
             round(np.mean(data_t[f]), 1),
             moda[0], moda[1],
             np.median(ds)))
    plot_boxnotch_univariateanalysis(data_t, features_name)
    return

#df=pd.read_csv("../dataset/interpolated/time/ARDR.csv", delimiter=',', header=0)
df=pd.read_csv("../data_acquisition/dataset/original/BTC.csv", delimiter=',', header=0)
# Converting the column to DateTime format
df.Date = pd.to_datetime(df.Date, format='%Y-%m-%d')
df = df.set_index('Date')
#log_scaling(df)
#lag_plot_mine(df)
#info_bivariate(df.values,df.columns.values)
"""df=df.drop(['Volume'],axis=1)
info_univariate(df.values,df.columns.values)"""
describe(df)
