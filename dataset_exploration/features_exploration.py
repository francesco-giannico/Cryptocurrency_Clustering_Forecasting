import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas
from scipy import stats
from scipy.stats import pearsonr
import seaborn as sns
from sklearn.preprocessing import StandardScaler


def info_bivariate(data, features_name):
    thre = 0.4
    d = np.array(data)
    data_t = np.transpose(d)
    el = np.arange(0, len(data_t))
    combo_index = list(itertools.product(el, repeat=2))
    fig3 = plt.figure(1, figsize=[200, 200], dpi=100, facecolor='w', edgecolor='black')
    i = 1
    for e in combo_index:
        ind1 = e.__getitem__(0)
        ind2 = e.__getitem__(1)
        c, t = pearsonr(data_t[ind1], data_t[ind2])
        titolo = '\n{} - {}\nP: --> {}'.format(features_name[ind1], features_name[ind2], round(c, 2))
        print(titolo)
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

"""dataframe = pandas.read_csv("crypto_preprocessing/step2_normalized/BTC_normalized.csv", delimiter=',',
                            header=0)
# data := lista di dati (ciascuna entry è a sua volta una lista)
data = dataframe.values

# X := lista di dati (ciascuna entry è l'insieme delle sole features di ciascuna entry)
a = dataframe.drop(columns=["DateTime", 'Symbol'])
print(a)
dd = a.values
col_name = a.columns.values
print(np.shape(data), "\n", col_name)
info_bivariate(dd, col_name)"""
#info_univariate(dd, col_name)

#sns.distplot(df_train['SalePrice']);

"""var = 'GrLivArea'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));"""
dataframe = pandas.read_csv("crypto_preprocessing/step0-5_data/MARS.csv", delimiter=',',
                            header=0)
# data := lista di dati (ciascuna entry è a sua volta una lista)
data = dataframe.values
df = dataframe.drop(columns=["DateTime", 'Symbol'])
dd = df.values
col_name = df.columns.values
sns.set()
sns.pairplot(df[col_name], size = 2.5)
#plt.show()

close_scaled = StandardScaler().fit_transform(df['Low'][:,np.newaxis])
low_range = close_scaled[close_scaled[:,0].argsort()][:10]
high_range= close_scaled[close_scaled[:,0].argsort()][-10:]
print('outer range (low) of the distribution:')
print(low_range)
print('\nouter range (high) of the distribution:')
print(high_range)

"""
#heatmap
import seaborn as sns
path_distance_matrix="crypto_clustering/distanceMeasures/distance_matrix_old.csv"
df = pd.read_csv(path_distance_matrix, delimiter=',', header=None).iloc[290:306,290:306]
sample = df.values
plt.figure(figsize = (7, 5))
sns.heatmap(sample, linecolor='white')
plt.title('DTW Distance Matrix')
plt.show()"""

"""df1 = pd.read_csv("crypto_preprocessing/step2_normalized/NTRN_normalized.csv", delimiter=',',header=0)
df = pd.read_csv("crypto_preprocessing/step2_normalized/CUBE_normalized.csv", delimiter=',',header=0)
df1 = df1.drop(columns=["DateTime", 'Symbol'])
df = df.drop(columns=["DateTime", 'Symbol'])
distance, path = fastdtw(df.values, df1.values, dist=euclidean)
print(distance)"""
