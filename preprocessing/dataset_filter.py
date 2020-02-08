import pandas as pd
from utility.reader import read_csv

def get_date_crypto_less_entry(folderpreprocessing,cluster):
    min=10000
    cryptoMin=''
    #dateTimeMin=pd.datetime.now()
    for id in cluster:
     for crypto in cluster[id]:
        csv= read_csv(folderpreprocessing+"/step2_normalized/"+ crypto +"_normalized.csv")
       # print(crypto+ " " +str(len(csv['Open'])))
        #ora conto quante entry ha il file
        length= len(csv['Open']) #conto le righe
        if length<=min:
           # dateTime=pd.to_datetime(csv["DateTime"][0])
            #if(dateTime<dateTimeMin):
            min = length
            #dateTimeMin = dateTime
            cryptoMin=crypto
       # print("cluster "+ str(id) + " la minima e': " + cryptoMin)
        #recupero la data piu' vecchia della criptovaluta con meno entry
    csv = read_csv(folderpreprocessing+"/step2_normalized/" + cryptoMin + "_normalized.csv")
    return csv['DateTime'][0]