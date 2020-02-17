import os
import pandas as pd

from preparation.data_cleaning import remove_uncomplete_rows_by_range, input_missing_values
from preparation.data_construction import normalize
from preparation.data_selection import find_by_dead_before, find_uncomplete,remove_features
from utility.folder_creator import folder_creator
from utility.reader import get_crypto_symbols

FEATURES=['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
PATH_RAW_DATA = "../acquisition/dataset/original/"
PATH_PREPROCESSED = "../preparation/preprocessed_dataset/"
#path = path_preprocessed+"/"+ "step_0/"

DIR_STEP_ZERO= "step_0"
DIR_STEP_ONE = "step_1"
DIR_STEP_TWO = "step_2"
#folder_step_one = "step1_indicators"

def preprocessing(type):
    CRYPTO_SYMBOLS= get_crypto_symbols()
    folders_setup()
    step_0(CRYPTO_SYMBOLS)
    #step_1()
    """if type=="indexes":
        step_additionalFeatures()
        step_normalization_indexes()
       else:"""
    # step_normalization_noindexes()

def folders_setup():
    # Set the name of folder in which to save all intermediate results
    #folder_creator(PATH_PREPROCESSED,1)
    """folder_creator(PATH_PREPROCESSED + "/" + DIR_STEP_ZERO,1)
    folder_creator(PATH_PREPROCESSED + "/" + DIR_STEP_ONE, 1)
    folder_creator(PATH_PREPROCESSED + "/" + DIR_STEP_TWO,1)"""
    #folder_creator(path_preprocessed + "/"+ folder_step_one,1) #indexes

# ------------------------------------------
# STEP.0: PreProcessData and delete the ones with the older date upper to 05-2016
# ------------------------------------------
def step_0(CRYPTO_SYMBOLS):
    """remove_features(["Volume"])
    find_by_dead_before()
    find_uncomplete()"""
    """remove_uncomplete_rows_by_range("ARDR","2016-01-01","2017-01-01")
    remove_uncomplete_rows_by_range("REP", "2015-01-01", "2017-01-01")
    #todo ricorda che LKK lo abbiamo rimosso perchè ha 144 missing values nel 2018!!
    input_missing_values()"""
    normalize()
    """#Converts data into our format
    output_indicators_path =  name_folder + "/" + folder_step_zero + "/"

    #lista degli elementi in step0_data, cambia il nome di ogni file
    for each_stock in os.listdir(raw_data):
        name = each_stock.replace(".csv", "")
        #recupero il nome del Coin
        if name in COINS:
            #prende i raw data di questo coin e li trasforma
            file = raw_data + each_stock
            preparation.generate_normal(file, output_indicators_path, name)"""

# ------------------------------------------
# STEP.0,5: Convert Values to USD and/or remove Volumes
# ------------------------------------------
# Convert Values to USD and/or remove Volumes
def step_1():
    USDBTC = []
    #deve per forza leggere questo per primo.
    csv = pd.read_csv(path + "BTC.csv")
    USDBTC = csv["Open"].values[::-1]

    for file in os.listdir(path):
        csv = pd.read_csv(path + file)
        del csv["Volume"]
        del csv["VolumeBTC"]
        if file != "BTC.csv":
            # Optional remove Volumes
            csv["Open"] = csv["Open"].values[::-1]
            csv["Open"] = csv["Open"] * USDBTC[:len(csv["Open"])]
            csv["Open"] = csv["Open"].values[::-1]

            csv["High"] = csv["High"].values[::-1]
            csv["High"] = csv["High"] * USDBTC[:len(csv["Open"])]
            csv["High"] = csv["High"].values[::-1]

            csv["Low"] = csv["Low"].values[::-1]
            csv["Low"] = csv["Low"] * USDBTC[:len(csv["Open"])]
            csv["Low"] = csv["Low"].values[::-1]

            csv["Close"] = csv["Close"].values[::-1]
            csv["Close"] = csv["Close"] * USDBTC[:len(csv["Open"])]
            csv["Close"] = csv["Close"].values[::-1]

        #csv.to_csv( name_folder + "/" + folder_step_half + "/" + file, index=False)


# ------------------------------------------
# STEP.1: Add Additional Features
# ------------------------------------------
# Listing all available time series (original data)
"""def step_additionalFeatures():

    output_indicators_path =  name_folder + "/" + folder_step_one + "/"
    # Execute over all time series in the folder chosen
    # Performs indicators: RSI, SMA, EMA
    # Over 14, 30, 60 previous days
    lookback = [14, 30, 60]
    path =  name_folder + "/" + folder_step_half + "/"
    for each_stock in os.listdir(path):
        name = each_stock.replace(".csv", "")
        #if name in COINS:
        file = path + each_stock
        preparation.generate_indicators(file, "Close", lookback, output_indicators_path, name)"""

# ------------------------------------------
# STEP.2: Normalize Data - no indexes
# ------------------------------------------
"""def step_normalization_noindexes():
    data =  name_folder + "/" + folder_step_half + "/"
    stock_series = os.listdir(data)
    output_normalized_path = name_folder + "/" + folder_step_two + "/"
    # Chosen features to exclude in normalizing process
    excluded_features = ['DateTime', 'Symbol']
    for each_stock in stock_series:
        name = each_stock.replace(".csv", "")
        file = data + "/" + each_stock
        #preparation.normalized(file, excluded_features, output_normalized_path, name)"""

# ------------------------------------------
# STEP.2: Normalize Data - indexes
# ------------------------------------------
"""def step_normalization_indexes():
        with_indicators_data =  name_folder + "/" + folder_step_one + "/"
        with_indicators_stock_series = os.listdir(with_indicators_data)
        output_normalized_path =  name_folder + "/" + folder_step_two + "/"
        # Chosen features to exclude in normalizing process
        excluded_features = ['DateTime', 'Symbol']
        for each_stock_with_indicators in with_indicators_stock_series:
            name = each_stock_with_indicators.replace("_with_indicators.csv", "")
            file = with_indicators_data + "/" + each_stock_with_indicators
            preparation.normalized(file, excluded_features, output_normalized_path, name)"""


# ------------------------------------------
# Create cut files from specified day for horizontal dataset
# ------------------------------------------
"""def cut_crypto(first_day,cluster,cluster_id,clustering_algorithm,type,folderoutput,preprocessingfolder):
    #quelli NON normalizzati, perchè tanto li normalizza la LSTM
    name_folder=preprocessingfolder
    if type=="indexes":
      folder_data = folder_step_one #indici
      end = "_with_indicators.csv"
    else:
      folder_data = folder_step_half
      end = ".csv"

    folder_creator("crypto_clustering/"+folderoutput+"/cutData", 0)
    folder_creator("crypto_clustering/"+folderoutput+"/cutData/" + clustering_algorithm, 0)
    #per tutte le crypto del cluster specifico
    for id in cluster:
        complete_path = "crypto_clustering/"+folderoutput+"/cutData/" + clustering_algorithm+ "/cluster_" + str(cluster_id)
        folder_creator(complete_path, 1)
        for crypto in cluster[id]:
            after_data = False
            fileToRead=open(name_folder+ "/" +folder_data + "/"+ crypto+end, "r")
            fileToWrite = open(complete_path+ "/" +crypto+".csv", "w")
            for line in fileToRead:
                if (line.startswith(first_day)):
                    after_data=True
                if(after_data or line.startswith("Date")):
                    fileToWrite.write(line)
            fileToRead.close()
            fileToWrite.close()

 # ------------------------------------------
#  Create horizontal dataset from cut files
# ------------------------------------------

def create_horizontal_from_cut(clustering_algorithm,cluster_id,type,folderoutput):
    folder_data =  "crypto_clustering/" + folderoutput + "/cutData/" + clustering_algorithm+"/cluster_"+str(cluster_id)
    folder_creator("crypto_clustering/"+ folderoutput+"/horizontalDataset", 0)
    folder_creator("crypto_clustering/"+ folderoutput+"/horizontalDataset/" + clustering_algorithm, 0)
    folder_creator("crypto_clustering/"+ folderoutput+"/horizontalDataset/" + clustering_algorithm+"/cluster_"+str(cluster_id), 1)
    filename = "/horizontal.csv"
    if type=="indexes":
        filename = "/horizontal_indicators.csv"
    file=[]
    primo=True
    n=0
    colonne=0
    #concatea in orizzontale tutti i file in un array di stringhe
    folder_horizontal= "crypto_clustering/"+ folderoutput+"/horizontalDataset/" + clustering_algorithm+"/cluster_"+str(cluster_id)
    for crypto in os.listdir(folder_data):
        n+=1
        fileToRead = open(folder_data+ "/" + crypto, "r")
        i = 0
        if primo:
            primo = False
            for line in fileToRead:
                if i==0:
                    colonne=len(line.split(","))
                file.append(line)
                i += 1
        else:
            for line in fileToRead:
                file[i]=file[i][:-1]+","+line
                split=file[i].split(",",1)
                split[1]=split[1].replace(split[0]+",","")
                file[i]=split[0]+","+split[1]
                i+=1
        fileToRead.close()

    #aggiungi un numero per disambiguare le colonne
    for i in range (n,0, -1):
        if i==n:
            file[0]=file[0].replace(",", "_" + str(i) + ",")
        else:
            if i==1:
                file[0]=file[0].replace("_"+str(i+1)+",", "_" + str(i) + ",", colonne)
            else:
                file[0]=file[0].replace("_"+str(i+1)+",", "_" + str(i) + ",", colonne+((i-1)*(colonne-1)))

    #rimuovi il numero per la prima colonna e aggiungilo all'ultima
    file[0]=file[0].replace("_1,", ",", 1)[:-1]+ "_"+str(n)+"\n"

    #scrivi l'array nel file
    fileToWrite = open(folder_horizontal + filename, "w")
    for line in file:
        fileToWrite.write(line)
    fileToWrite.close()"""