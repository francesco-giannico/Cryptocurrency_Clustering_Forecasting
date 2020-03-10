import os
import pandas as pd
import pandas_ta as panda

from utility.folder_creator import folder_creator

PATH_NORMALIZED_FOLDER="../preparation/preprocessed_dataset/constructed/normalized/"
PATH_INTEGRATED_FOLDER="../preparation/preprocessed_dataset/integrated/"
FEATURE="Close"
LOOKBACK = [14, 30, 60]

def integrate_with_indicators(input_path,type):
    folder_creator(PATH_INTEGRATED_FOLDER+type,1)
    for crypto in os.listdir(input_path):
        df = pd.read_csv(input_path+crypto, sep=',',header=0)
        df["Date"] = pd.to_datetime(df["Date"])

        #df = df.sort_values('Date', ascending=True)
        data_series_of_feature = df[FEATURE]
        for lookback_value in LOOKBACK:
            #df[str('RSI_' + str(lookback_value))] = get_RSI(data_series_of_feature,lookback_value)
            df[str('SMA_' + str(lookback_value))] = get_SMA(data_series_of_feature,lookback_value)
            df[str('EMA_' + str(lookback_value))] = get_EMA(data_series_of_feature,lookback_value)
        df.fillna(value=0, inplace=True)
        df.to_csv(PATH_INTEGRATED_FOLDER+type+"/"+crypto,sep=",", index=False)


def get_RSI(data_series_of_feature,lookback_value):
   return panda.rsi(data_series_of_feature, length=lookback_value)

def get_SMA(data_series_of_feature,lookback_value):
    return panda.sma(data_series_of_feature, length=lookback_value)

def get_EMA(data_series_of_feature,lookback_value):
    return panda.ema(data_series_of_feature, length=lookback_value)

