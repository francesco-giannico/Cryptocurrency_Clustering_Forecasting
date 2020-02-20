import pandas as pd

#it takes only the slice of the dataframe you are interested in
def cut_dataset_by_range(PATH,crypto_symbol,start_date,end_date):
    df = pd.read_csv(PATH + crypto_symbol + ".csv", delimiter=',', header=0)
    df = df.set_index("Date")
    df1=df.loc[start_date:end_date,:]
    df1 = df1.reset_index()
    return df1

