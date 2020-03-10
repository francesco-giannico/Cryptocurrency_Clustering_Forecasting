import itertools
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import lag_plot
from scipy.stats import pearsonr, stats
from understanding.missing_values import  count_missing_values, count_missing_values_by_year, \
    generate_bar_chart_by_year
from utility.folder_creator import folder_creator
from statsmodels.tsa.stattools import adfuller

COLUMNS=['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
PATH_DATA_UNDERSTANDING= "../understanding/output/"


"""few statistics that give some perspective on the nature of the distribution of the data.
-count the larger this number, the more credibility all the stats have.
-mean is the average and is the "expected" value of the distribution. On average, you'd expect to get this number.
it's affected by outliers.
-std (how a set of values spread out from their mean).
A low SD shows that the values are close to the mean and a high SD shows a high diversion from the mean.
SD is affected by outliers as its calculation is based on the mean
If SD is zero, all the numbers in a dataset share the same value
-50% is also the median and it's difference from the mean gives information on the skew of the distribution. It's also another definition of average that is robust to outliers in the data.
-25% percentile is the value below which 25% of the observations may be found
-75%  percentile is the value below which 75% of the observations may be found
-min, max, max - min, 75% - 25% are all alternatives to perspectives on how big of swings the data takes relative to the mean
"""
import seaborn as sns
def describe(PATH_DATASET,output_path,name_folder_res,features_to_use):
    PATH_OUT=output_path+"descriptions/"+name_folder_res+"/"
    folder_creator(PATH_OUT,1)
    for crypto in os.listdir(PATH_DATASET):
        crypto_name = crypto.replace(".csv", "")
        df = pd.read_csv(PATH_DATASET + crypto, delimiter=',', header=0,usecols=features_to_use)
        if(crypto_name=="ADA"):
            PATH_CRYPTO=PATH_OUT+crypto_name+"/"
            for feature in features_to_use:
                # mean,quantile ecc..
                folder_creator(PATH_CRYPTO + "some_stats/",0)
                df.describe().to_csv(PATH_CRYPTO + "some_stats/"+crypto, sep=",")

                folder_creator(PATH_CRYPTO + "noscaling_vs_logscaling/", 0)
                no_scaling_vs_log_scaling(df,feature,crypto_name,PATH_CRYPTO+"noscaling_vs_logscaling/")

                folder_creator(PATH_CRYPTO + "lag_plot/", 0)
                lag_plott(df,feature,crypto_name,PATH_CRYPTO + "lag_plot/")

                folder_creator(PATH_CRYPTO + "box_plot/", 0)
                box_plot(df,feature,crypto_name,PATH_CRYPTO + "box_plot/")

                folder_creator(PATH_CRYPTO + "distribution_plot/", 0)
                distribution_plot(df,feature,crypto_name,PATH_CRYPTO + "distribution_plot/")

            folder_creator(PATH_CRYPTO + "stationary_test/", 0)
            stationary_test(df,features_to_use,crypto_name,PATH_CRYPTO + "stationary_test/")

def no_scaling_vs_log_scaling(df,feature_to_use,crypto_name,output_path):
    df = df.dropna(subset=[feature_to_use])
    plt.figure(figsize=(20, 6))
    plt.subplot(1, 2, 1)
    ax = df[feature_to_use].plot(style=['-'])
    ax.lines[0].set_alpha(0.3)
    ax.set_ylim(-0.01, np.max(df[feature_to_use]))
    plt.xticks(rotation=90)
    plt.title("No scaling")
    ax.legend()

    plt.subplot(1, 2, 2)
    ax = df[feature_to_use].plot(style=['-'])
    ax.lines[0].set_alpha(0.3)
    ax.set_yscale('log')
    #ax.set_ylim(0, np.max(df['Close']+1))
    plt.xticks(rotation=90)
    plt.title("logarithmic scale")
    ax.legend()
    plt.savefig(output_path+crypto_name+"_"+feature_to_use+".png",dpi=120)

def lag_plott(df,feature,crypto_name,output_path):
    df = df.dropna(subset=[feature])
    plt.figure(figsize=(5,5))
    plt.title("lag_plot_"+feature+"_"+crypto_name)
    lag_plot(df[feature])
    plt.savefig(output_path + crypto_name + "_" + feature + ".png", dpi=120)

def box_plot(df,feature,crypto_name,output_path):
    plt.figure(figsize=(5, 5))
    ax = sns.boxplot(x=feature,data=df, orient="v")
    ax.set_title("box_plot_" + feature + "_" + crypto_name)
    plt.savefig(output_path + crypto_name + "_" + feature + ".png", dpi=120)

def distribution_plot(df,feature,crypto_name,output_path):
    filter_data = df.dropna(subset=[feature])
    plt.figure(figsize=(14, 8))
    ax=sns.distplot(filter_data[feature], kde=False)
    ax.set_title("distribution_plot_" + feature + "_" + crypto_name)
    plt.savefig(output_path + crypto_name + "_" + feature + ".png", dpi=120)

def stationary_test(df,features,crypto_name,output_path):
    significance_level=0.05
    res = {'feature': [], 'adf_statistics': [], 'p-value': [],'1%':[],'5%':[],'10%':[],'is_stationary':[]}
    for feature in features:
        df = df.dropna(subset=[feature])
        X = df[feature].values
        result = adfuller(X,autolag='AIC')
        res["adf_statistics"].append(float(result[0]))
        p_value=result[1]
        res['p-value'].append(p_value)
        res['feature'].append(feature)
        for key, value in result[4].items():
          res[key]=value
        if (p_value < significance_level):
            res['is_stationary'].append('True')
        else:
            res['is_stationary'].append('False')
    pd.DataFrame(data=res).to_csv(output_path + crypto_name +".csv",sep=",",index=False)

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

"""#df=pd.read_csv("../dataset/interpolated/time/ARDR.csv", delimiter=',', header=0)
df=pd.read_csv("../acquisition/dataset/original/BTC.csv", delimiter=',', header=0)
# Converting the column to DateTime format
df.Date = pd.to_datetime(df.Date, format='%Y-%m-%d')
df = df.set_index('Date')
#log_scaling(df)
#lag_plot_mine(df)
#info_bivariate(df.values,df.columns.values)
df=df.drop(['Volume'],axis=1)
info_univariate(df.values,df.columns.values)
describe(df)"""
