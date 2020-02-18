import csv
import json
import numpy as np
from utility.folder_creator import folder_creator


def save_clusters(clusters,filename,folderoutput):
    # leggo il dizionario: crypto-id
    cryptoSymbol_id = read_json('crypto_clustering/distanceMeasures/cryptoSymbol_id.json')
    cluster_cyptocurrencies=[]

    fileToWrite = open("crypto_clustering/"+folderoutput+"/clusters/"+ filename+".json", "w")
    fileToWrite2 = open("crypto_clustering/"+folderoutput+"/clusters/" + filename + ".txt", "w")
    i=0
    for cluster in clusters:
        cryptocurrencies = []
        for crypto_id in cluster:
            cryptocurrencies.append(cryptoSymbol_id[str(crypto_id)])
        cluster_cyptocurrencies.append({str(i):cryptocurrencies})
        line= "CLUSTER_"+str(i)+ "\n" +str(cryptocurrencies)
        fileToWrite2.write(line+"\n\n")
        i += 1
    json.dump(cluster_cyptocurrencies, fileToWrite)
    fileToWrite.close()
    fileToWrite2.close()

def save_dummy_clusters(filename,folderoutput):
    # leggo il dizionario: crypto-id
    cryptoSymbol_id = read_json('crypto_clustering/distanceMeasures/cryptoSymbol_id.json')
    cluster_cyptocurrencies=[]

    fileToWrite = open("crypto_clustering/"+folderoutput+"/clusters/"+ filename+".json", "w")
    #cryptocurrencies = []
    cluster_cyptocurrencies.append({str(0):cryptoSymbol_id})
    json.dump(cluster_cyptocurrencies, fileToWrite)
    fileToWrite.close()


def save_clusters_agglomerative (labels,filename):
    # leggo il dizionario: crypto-id
    cryptoSymbol_id = read_json('crypto_clustering/distanceMeasures/cryptoSymbol_id.json')
    cluster_cyptocurrencies={}
    tatan=[]
    fileToWrite = open("crypto_clustering/results/"+ filename+".json", "w")
    fileToWrite2 = open("crypto_clustering/results/" + filename + ".txt", "w")

    for label in np.sort(labels):
        cluster_cyptocurrencies[str(label)]=[]

    i = 0
    for label in labels:
        cluster_cyptocurrencies[str(label)].append(cryptoSymbol_id[str(i)])
        """if cryptoSymbol_id[str(i)] == "BTC":
            print("BTC" + " in cluster " + str(label))
        if cryptoSymbol_id[str(i)] == "LTC":
            print("LTC" + " in cluster " + str(label))
        if cryptoSymbol_id[str(i)] == "DOGE":
            print("DOGE" + " in cluster " +str(label))
        if cryptoSymbol_id[str(i)] == "DASH":
            print("DASH" + " in cluster " + str(label))
        if cryptoSymbol_id[str(i)] == "XLM":
            print("XLM" + " in cluster " + str(label))
        if cryptoSymbol_id[str(i)] == "XRP":
            print("XRP" + " in cluster " + str(label))
        if cryptoSymbol_id[str(i)] == "XMR":
            print("XMR" + " in cluster " + str(label))
        if cryptoSymbol_id[str(i)] == "XEM":
            print("XEM" + " in cluster " + str(label))"""
        """if cryptoSymbol_id[str(i)] == "ETH":
            print("ETH" + " in cluster " + str(label))
        if cryptoSymbol_id[str(i)] == "ETC":
            print("ETC" + " in cluster " + str(label))"""
        i+=1


    for c in cluster_cyptocurrencies:
         line = "CLUSTER_" + str(c)+ "\n" +str(cluster_cyptocurrencies[c])+ "\n"
         fileToWrite2.write(line + "\n\n")

    tatan.append(cluster_cyptocurrencies)
    json.dump(tatan, fileToWrite)
    fileToWrite.close()
    fileToWrite2.close()



def save_clusters2(clusters,filename):
    # leggo il dizionario: crypto-id
    cryptoSymbol_id = read_json('crypto_clustering/distanceMeasures/cryptoSymbol_id.json')
    cluster_cyptocurrencies=[]

    fileToWrite = open("crypto_clustering/results/"+ filename+".json", "w")
    fileToWrite2 = open("crypto_clustering/results/" + filename + ".txt", "w")
    i=0
    for label in clusters:
        cryptocurrencies = []
        for crypto_id in clusters[label]:
            cryptocurrencies.append(cryptoSymbol_id[str(crypto_id)])

            """" if cryptoSymbol_id[str(crypto_id)] == "BTC":
                print("BTC" + " in cluster " + str(i))
            if cryptoSymbol_id[str(crypto_id)] == "LTC":
                print("LTC" + " in cluster " + str(i))
            if cryptoSymbol_id[str(crypto_id)] == "DOGE":
                print("DOGE" + " in cluster " + str(i))
            if cryptoSymbol_id[str(crypto_id)] == "DASH":
                print("DASH" + " in cluster " + str(i))
            if cryptoSymbol_id[str(crypto_id)] == "XLM":
                print("XLM" + " in cluster " + str(i))
            if cryptoSymbol_id[str(crypto_id)] == "XRP":
                print("XRP" + " in cluster " + str(i))
            if cryptoSymbol_id[str(crypto_id)] == "XMR":
                print("XMR" + " in cluster " + str(i))
            if cryptoSymbol_id[str(crypto_id)] == "XEM":
                print("XEM" + " in cluster " + str(i))"""
            """ if cryptoSymbol_id[str(crypto_id)] == "ETH":
                print("ETH" + " in cluster " + str(i))
            if cryptoSymbol_id[str(crypto_id)] == "ETC":
                print("ETC" + " in cluster " + str(i))"""

        cluster_cyptocurrencies.append({str(i):cryptocurrencies})
        line= "CLUSTER_"+str(i)+ "\n" +str(cryptocurrencies)
        fileToWrite2.write(line+"\n\n")
        i += 1
    json.dump(cluster_cyptocurrencies, fileToWrite)
    fileToWrite.close()
    fileToWrite2.close()


def save_distance_matrix(distance_matrix,filename):
    folder_creator("../modelling/techniques/clustering/distance_measures/",1)
    with open("../modelling/techniques/clustering/distance_measures/"+filename +'.csv', 'w') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerows(distance_matrix)
    writeFile.close()

def save_dict_symbol_id(dict):
    with open('../modelling/techniques/clustering/symbol_id.json', 'w') as fp:
        json.dump(dict, fp)
    fp.close()