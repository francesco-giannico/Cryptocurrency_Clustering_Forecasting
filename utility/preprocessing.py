import pandas as pd
import pandas_ta as panda
from sklearn.preprocessing import MinMaxScaler


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
        series_rsi = panda.rsi(data_series_of_feature, length=lookback_value)
        data[str('RSI_' + str(lookback_value))] = series_rsi
        series_sma = panda.sma(data_series_of_feature, length=lookback_value)
        data[str('SMA_' + str(lookback_value))] = series_sma
        series_ema = panda.ema(data_series_of_feature, length=lookback_value)
        data[str('EMA_' + str(lookback_value))] = series_ema
    data.fillna(value=0, inplace=True)
    data.to_csv(output_path + filename_output + "_with_indicators" + ".csv", index=False)
    return


def normalized(filepath, features_to_exclude, output_path, filename_output):
    data = pd.read_csv(filepath, sep=',')
    scaler = MinMaxScaler()
    for col in data.columns:
        if col not in features_to_exclude:
            normalized = scaler.fit_transform(data[col].values.reshape(-1, 1))
            data[col] = pd.Series(normalized.reshape(-1))
    data.to_csv(output_path + filename_output + "_normalized.csv", index=False)
    return


def generate_normal(filepath, output_path, filename_output):
    data = pd.read_csv(filepath, sep=',')
    if filepath.count("BTC") > 0:
        data["DateTime"] = pd.to_datetime(data["DateTime"], dayfirst=True)
        data = data.sort_values('DateTime', ascending=True)
    else:
        data["DateTime"] = pd.to_datetime(data["DateTime"])
        data = data.sort_values('DateTime', ascending=True)

    data.fillna(value=0, inplace=True)
    #rimuovo quelle nate dopo il  1-05-2016
    dateTimeMin=pd.to_datetime("30-11-2014")
    length = len(data["DateTime"])
    dateTime=''
    if length > 0:
        if filepath.count("BTC")>0:
            dateTime = pd.to_datetime(data["DateTime"][0])
        else:
          dateTime = pd.to_datetime(data["DateTime"][length - 1])
        if dateTime <= dateTimeMin:
            data.to_csv(output_path + filename_output + ".csv", index=False)
    return