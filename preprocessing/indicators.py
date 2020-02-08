import pandas as pd
import pandas_ta as panda

def generate_indicators(filepath, name_feature, lookback_list, output_path, filename_output):
    data = pd.read_csv(filepath, sep=',')
    data["DateTime"] = pd.to_datetime(data["DateTime"])
    data = data.sort_values('DateTime', ascending=True)
    '''
    if filepath.count("BTC") > 0:
        data["DateTime"] = pd.to_datetime(data["DateTime"], dayfirst=True)
        data = data.sort_values('DateTime', ascending=True)
    else:
        data["DateTime"] = pd.to_datetime(data["DateTime"])
        data = data.sort_values('DateTime', ascending=True)
    '''
    #data = data.sort_values('DateTime')
    data_series_of_feature = data[name_feature]
    for lookback_value in lookback_list:
        data[str('RSI_' + str(lookback_value))] = get_RSI(data_series_of_feature,lookback_value)
        data[str('SMA_' + str(lookback_value))] = get_SMA(data_series_of_feature,lookback_value)
        data[str('EMA_' + str(lookback_value))] = get_EMA(data_series_of_feature,lookback_value)
    data.fillna(value=0, inplace=True)
    data.to_csv(output_path + filename_output + "_with_indicators" + ".csv", index=False)
    return


def get_RSI(data_series_of_feature,lookback_value):
   return panda.rsi(data_series_of_feature, length=lookback_value)

def get_SMA(data_series_of_feature,lookback_value):
    return panda.sma(data_series_of_feature, length=lookback_value)

def get_EMA(data_series_of_feature,lookback_value):
    return panda.ema(data_series_of_feature, length=lookback_value)