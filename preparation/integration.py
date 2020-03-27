import os
import pandas as pd
import pandas_ta as panda

from utility.folder_creator import folder_creator

PATH_TRANSFORMED_FOLDER= "../preparation/preprocessed_dataset/transformed/"
PATH_INTEGRATED_FOLDER= "../preparation/preprocessed_dataset/integrated/"
FEATURE="Close"

#Loockbacks extracted from cryptocompare, in the chart section filters
LOOKBACK_RSI =[14,21]
LOOKBACK_EMA=[5,12,26,50]
LOOKBACK_SMA=[5,13,20,30,50]

def integrate_with_indicators(input_path):
    folder_creator(PATH_INTEGRATED_FOLDER,1)
    for crypto in os.listdir(input_path):
        df = pd.read_csv(input_path+crypto, sep=',',header=0)
        df["Date"] = pd.to_datetime(df["Date"])

        #df = df.sort_values('Date', ascending=True)
        data_series_of_feature = df[FEATURE]
        for lookback_value in LOOKBACK_RSI:
            df[str('RSI_' + str(lookback_value))] = get_RSI(data_series_of_feature,lookback_value)
        for lookback_value in LOOKBACK_SMA:
            df[str('SMA_' + str(lookback_value))] = get_SMA(data_series_of_feature,lookback_value)
        for lookback_value in LOOKBACK_EMA:
            df[str('EMA_' + str(lookback_value))] = get_EMA(data_series_of_feature,lookback_value)

        df['lag_1'] = df['Close'].shift(1)
        df['lag_7'] = df['Close'].shift(7)
        df = df.iloc[7:]

        df.fillna(value=0, inplace=True)
        df.to_csv(PATH_INTEGRATED_FOLDER+"/"+crypto,sep=",", index=False)




def get_RSI(data_series_of_feature,lookback_value):
   return panda.rsi(data_series_of_feature, length=lookback_value)

def get_SMA(data_series_of_feature,lookback_value):
    return panda.sma(data_series_of_feature, length=lookback_value)

def get_EMA(data_series_of_feature,lookback_value):
    return panda.ema(data_series_of_feature, length=lookback_value)

def integrate_with_lag(input_path):
    folder_creator(PATH_INTEGRATED_FOLDER, 1)
    for crypto in os.listdir(input_path):
        df = pd.read_csv(input_path + crypto, sep=',', header=0)
        df["Date"] = pd.to_datetime(df["Date"])
        df['lag_1'] = df['Close'].shift(1)
        df['lag_2'] = df['Close'].shift(2)
        df['lag_3'] = df['Close'].shift(3)
        df['lag_7'] = df['Close'].shift(7)
        df = df.iloc[7:]
        df.to_csv(PATH_INTEGRATED_FOLDER + "/" + crypto, sep=",", index=False)
