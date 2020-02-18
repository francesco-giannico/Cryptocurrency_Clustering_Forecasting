from preparation.cleaning import remove_uncomplete_rows_by_range, input_missing_values
from preparation.construction import normalize
from preparation.integration import integrate_with_indicators
from preparation.selection import find_by_dead_before, find_uncomplete,remove_features
from utility.folder_creator import folder_creator

PATH_PREPROCESSED = "../preparation/preprocessed_dataset/"

def preprocessing(type):
    folders_setup()
    feature_selection()
    separation()
    cleaning()
    construction()
    integration()

def folders_setup():
    # Set the name of folder in which to save all intermediate results
    folder_creator(PATH_PREPROCESSED,0)

def feature_selection():
    remove_features(["Volume"])

def separation():
    find_by_dead_before()
    find_uncomplete()

def cleaning():
    remove_uncomplete_rows_by_range("ARDR","2017-01-01","2019-12-31")
    remove_uncomplete_rows_by_range("REP", "2017-01-01", "2019-12-31")
    #todo ricorda che LKK lo abbiamo rimosso perchè ha 144 missing values nel 2018!!
    input_missing_values()

def construction():
    #feature scaling
    normalize()

def integration():
    integrate_with_indicators()

#todo manca l'horizontal dataset. Poi vediamo
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