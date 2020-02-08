import pandas as pd


"""def generate_normal(filepath, output_path, filename_output):
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
    return"""