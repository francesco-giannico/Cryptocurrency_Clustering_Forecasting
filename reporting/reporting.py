import matplotlib.pyplot as plt
import numpy as np
import os
from itertools import product

import numpy
import pandas as pd

from crypto_utility.folder_creator import folder_creator
from crypto_utility.reader import get_cryptocurrenciesSymbols, read_json, get_clusters, get_clusters2


def overall_report(title,filenameout,filenamein):
    pathGeneric = "crypto_clustering/reports"
    name = "/"+filenamein
    csv = pd.read_csv(pathGeneric + "/" + name + ".csv")
    ks_x = []

    exp_name = ["dtw", "pearson","multitarget","singletarget"]

    yvals = []
    zvals = []
    gvals=[]
    mvals=[]
    ks=['sqrtN',"sqrtNdiv2","sqrtNby2","sqrtNdiv4","sqrtNby4"]

    tofind = "multitarget"
    filt1 = csv[csv["k_description"] == tofind]
    gvals.append(filt1['average_rmse_norm'].values[0])

    tofind = "singletarget"
    filt1 = csv[csv["k_description"] == tofind]
    mvals.append(filt1['average_rmse_norm'].values[0])

    for k in ks:
        tofind="dtw_"+k
        tofind2="pearson_"+k
        filt1=csv[csv["k_description"]==tofind]
        filt2=csv[csv["k_description"]==tofind2]
        yvals.append(filt1['average_rmse_norm'].values[0])
        zvals.append(filt2['average_rmse_norm'].values[0])
        #print(filt1['k'].values[0])
        ks_x.append(k+"="+str(filt1['k'].values[0]))


    ind = np.arange(len(ks_x))  # the x locations for the groups
    ind = ind * 2.3
    width = 0.30 # the width of the bars

    fig = plt.figure(figsize=(25, 10), dpi=70)
    ax = fig.add_subplot(111)
    ax.set_title(title)

    rects1 = ax.bar(ind, yvals, width, color='green')
    rects2 = ax.bar(ind + width, zvals, width, color='orange')
    rects3 = ax.bar(ind + width+width, gvals, width, color='blue')
    rects4 = ax.bar(ind + width + width+width, mvals, width, color='olive')
    def autolabel(rects):
        #Attach a text label above each bar displaying its height

        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2., 1 * height,
                    "%.6f" % float(height),
                    ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    autolabel(rects4)
    ax.set_ylabel('Average (RMSE)')
    ax.set_xlabel('Number of clusters (k)')
    ax.set_xticks(ind)
    ax.set_xticklabels(ks_x)

    ax.legend((rects1[0],rects2[0],rects3[0],rects4[0]), (exp_name[0],exp_name[1],exp_name[2],exp_name[3]))
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.grid(linewidth=0.2, color='black')
    plt.savefig(pathGeneric + "/"+filenameout+".png")
    # plt.show()"""


def barchartComparison(experiments):
    pathGeneric = "crypto_clustering/experiment_common/"
    name = "/myresults.csv"
    csvs=[]

    for folder in os.listdir(pathGeneric):
        if (folder != "clusters" and folder != "cutData" and folder != "horizontalDataset"):
            csvs.append(pd.read_csv(pathGeneric + folder +name, usecols=['cluster_id', 'crypto_name', 'average_rmse_norm']))


    for experiment in experiments:
     if experiment!="experiment_common":
         pathGeneric= "crypto_clustering/"+experiment+"/"
         folder_creator(pathGeneric+"reports",1)
         for folder in os.listdir(pathGeneric):
             name = "/myresults.csv"
             if (folder!="clusters" and folder!="cutData" and folder!="horizontalDataset" and folder!="reports"):
                csvs.append(pd.read_csv(pathGeneric+folder+name,usecols=['cluster_id','crypto_name','average_rmse_norm']))
         ks_x=[]
         i = 0
         while i < len(csvs[0]["crypto_name"]):
            ks_x.append(str(csvs[0]["crypto_name"][i]))
            i += 1

         ind = np.arange(len(ks_x))  # the x locations for the groups
         ind = ind*1.5
         width = 0.15 # the width of the bars

         fig = plt.figure(figsize=(30, 10), dpi=150)
         ax = fig.add_subplot(111)
         ax.set_title("Comparision: Single target VS Multitarget (NO INDICATORS)")

         csv1V = []
         csv2V = []
         csv3V=[]
         csv4V=[]
         csv5V=[]
         csv6V=[]
         csv7V=[]
         for name in ks_x:
          crypto1= csvs[0][csvs[0]["crypto_name"]==name]
          crypto2 = csvs[1][csvs[1]["crypto_name"] == name]
          crypto3 = csvs[2][csvs[2]["crypto_name"] == name]
          crypto4 = csvs[3][csvs[3]["crypto_name"] == name]
          crypto5 = csvs[4][csvs[4]["crypto_name"] == name]
          crypto6= csvs[5][csvs[5]["crypto_name"] == name]
          crypto7= csvs[6][csvs[6]["crypto_name"] == name]
          csv1V.append(crypto1["average_rmse_norm"].values[0])
          csv2V.append(crypto2["average_rmse_norm"].values[0])
          csv3V.append(crypto3["average_rmse_norm"].values[0])
          csv4V.append(crypto4["average_rmse_norm"].values[0])
          csv5V.append(crypto5["average_rmse_norm"].values[0])
          csv6V.append(crypto6["average_rmse_norm"].values[0])
          csv7V.append(crypto7["average_rmse_norm"].values[0])

         rects1 = ax.bar(ind, csv1V, width, color='green')
         rects2 = ax.bar(ind + width,  csv2V, width, color='orange')
         space= width*2
         rects3 = ax.bar(ind+space, csv3V, width, color='brown')
         space= width*3
         rects4 = ax.bar(ind + space,csv4V, width, color='blue')
         space= width*4
         rects5 = ax.bar(ind + space, csv5V, width, color='cyan')
         space= width*5
         rects6 = ax.bar(ind + space, csv6V, width, color='olive')
         space = width * 6
         rects7 = ax.bar(ind + space, csv7V, width, color='magenta')

         ax.set_ylabel('Average (RMSE)')
         ax.set_xlabel('Crypto names')
         ax.set_xticks(ind)
         ax.set_xticklabels(ks_x, rotation=45, ha="right")
         #ax.legend((rects1[0], rects2[0]), (algorithms[0], algorithms[1]))
         #ax.legend((rects1[0], rects2[0], rects3[0]),(algorithms[0], algorithms[1], algorithms[2]))
         # ax.legend((rects1[0],rects2[0],rects3[0],rects4[0]), (algorithms[0],algorithms[1],algorithms[2],algorithms[3]))

        #ax.legend((rects1[0],rects2[0],rects3[0],rects4[0], rects5[0]), (algorithms[0],algorithms[1],algorithms[2],algorithms[3],algorithms[4]))

         ax.legend((rects1[0],rects2[0],rects3[0],rects4[0], rects5[0], rects6[0],rects7[0]), ("multitarget","singletarget","sqrtN","sqrtNby2","sqrtNby4","sqrtNdiv2","sqrtNdiv4"))
         ax.set_axisbelow(True)
         plt.tight_layout()
         plt.grid(linewidth=0.3, color='black')
         plt.savefig(pathGeneric + "reports/comparison_singleVSmulti.png")
         # plt.show()


def barChart(path,pathToSave):
    final_csv = pd.read_csv(path)
    ks_x = []
    i = 0
    while i < len(final_csv["crypto_name"]):
        ks_x.append(str(final_csv["crypto_name"][i]))
        i += 1

    ind = np.arange(len(ks_x))  # the x locations for the groups
    ind = ind * 1.7
    width = 0.2  # the width of the bars

    fig = plt.figure(figsize=(20, 10), dpi=70)
    ax = fig.add_subplot(111)
    # ax.set_title('Cluster '+ str(clusterid)+": Average RMSE configuration oriented")

    yvals = []
    for avgrmse in final_csv["average_rmse_norm"].values:
        yvals.append(avgrmse)
    rects1 = ax.bar(ind, yvals, width, color='green')


    # zvals = [0.0600, 0.0522]
    # rects2 = ax.bar(ind + width, zvals, width, color='orange')

    ax.set_ylabel('Average (RMSE)')
    ax.set_xlabel('Crypto names')
    ax.set_xticks(ind)
    ax.set_xticklabels(ks_x, rotation = 90, ha="right")

    # ax.legend((rects1[0],rects2[0]), (exp_name[0],exp_name[1]))
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.grid(linewidth=0.2, color='black')
    plt.savefig(pathToSave + "/cluster.png")
    # plt.show()"""


def results_by_cluster(path,algorithms):
    for algorithm_k in algorithms:
        try:
            clusters = get_clusters(algorithm_k)
            N= len(clusters)
        except:
            N= 1

        df = pd.read_csv(path+ "/" + algorithm_k + "/myresults.csv",usecols=["cluster_id","crypto_name","average_rmse_norm"])
        dfOriginal=df
        for clusterID in range(N):
            partial= df[df['cluster_id']==clusterID]
            """avg = np.average(partial['average_rmse_norm'])
            partial = partial[partial['cluster_id'] ]"""
            partial = partial.sort_values(by=['average_rmse_norm'], ascending=False)
            pathFrom=path + "/" + algorithm_k + "/cluster_" + str(clusterID) + "/ordered_list_by_rmse.csv"
            partial.to_csv(pathFrom)
            barChart(pathFrom,path + "/" + algorithm_k+ "/cluster_" + str(clusterID))



        try:
            partial1 = dfOriginal.sort_values(by=['average_rmse_norm'], ascending=False)
            partial1.to_csv(path + "/" + algorithm_k + "/ordered_list_by_rmse_all.csv")
            pathFrom = path + "/" + algorithm_k +  "/ordered_list_by_rmse_all.csv"
            barChart(pathFrom, path + "/" + algorithm_k)
        except:
         pass
        """max=np.max(partial['average_rmse_norm'])
        min = np.min(partial['average_rmse_norm'])"""
        """avg = np.average(partial['average_rmse_norm'])
        listMax = partial[partial['average_rmse_norm'] >= avg]
        listMin = partial[partial['average_rmse_norm'] < avg]
        # avg= np.average(listMax['average_rmse_norm'])
        print(min)
        print(len(listMax))
        print(len(listMin))
        print(avg)
        listMax=listMax.sort_values(by=['average_rmse_norm'], ascending=False)
        listMax.to_csv(path+ "/" + algorithm_k + "/cluster_"+ str(clusterID) + "/list_max.csv")
        #rimuovo l'errore piu alto"""



def generate_fileaverageRMSE_byalgorithm(path,name_final_file,experiments):
    try:
        os.remove(path + "/" +name_final_file+".csv")
    except:
        pass
    all = {"k": [], "k_description": [], "average_rmse_norm": []}
    for experiment in experiments:
      for algorithm in os.listdir("crypto_clustering/" + experiment + "/"):
        if (algorithm!="clusters" and algorithm!="cutData" and algorithm!="horizontalDataset" and algorithm!="reports"):
            #all["k"].append(len(os.listdir(path+"/"+algorithm_k))-1)
            try:
              clusters = get_clusters2(experiment,algorithm)
              all["k"].append(len(clusters))
              if (experiment == "experiment_common"):
                all["k_description"].append(algorithm.split("_")[2])
              else:
                  name=experiment.split("_")[1]
                  all["k_description"].append(name+"_"+algorithm.split("_")[2])
            except:
                all["k"].append(0)
                all["k_description"].append("singleTarget")
            algorithm_results = pd.read_csv("crypto_clustering/"+experiment + "/"+algorithm +"/myresults.csv",usecols=["average_rmse_norm"])
            rmse=[]
            for rmseval in algorithm_results.values:
                rmse.append(rmseval)
            average=np.average(rmse)
            all["average_rmse_norm"].append(average)
    pd.DataFrame(all).to_csv(path+"/"+name_final_file+".csv")




def generate_averagermseForK(path,num_of_clusters,name_experiment_model):
    #print(num_of_clusters)
    try:
        os.remove(path + "/" +"myresults.csv")
    except:
        pass
    name_folder_result="Result"
    i=0
    #nota:average_rmse_norm è la media di tutti gli errori per giorni e hidden neurons
    all = {"cluster_id": [], "crypto_name": [], "average_rmse_norm": []}
    while i < num_of_clusters:
     if (name_experiment_model=="MultiTarget_Data"):
      completePath=path+"/"+"cluster_"+str(i)+"/"+name_experiment_model+"/"+name_folder_result +"/"
     else:
         completePath = path +  "/cluster_0/" + name_experiment_model + "/" + name_folder_result + "/"
     cryptocurrencies = os.listdir(completePath)
     for crypto in cryptocurrencies:
         all["cluster_id"].append(str(i))
         all["crypto_name"].append(crypto)
         configuration_used = os.listdir(completePath + crypto + "/")
         configuration_used.sort(reverse=True)
         # for each configuration:
         rmse=[]
         for conf in configuration_used:
             #estra solo il valore dell'rmse
             rmseval = pd.read_csv(completePath +str(crypto) + "/" + conf + "/stats/errors.csv",usecols=["rmse_norm"]).values[0][0]
             rmse.append(rmseval)
         average_rmse= np.average(rmse)
         all["average_rmse_norm"].append(average_rmse)
     i+=1
    pd.DataFrame(all).to_csv(path+"/myresults.csv")
    return


def join_predictions(name_folder_experiment, name_folder_result_experiment):
    try:
        os.remove(name_folder_experiment + "/" + name_folder_result_experiment + "/joined_predictions.csv")
    except:
        pass
    cryptocurrencies = os.listdir(name_folder_experiment + "/" + name_folder_result_experiment + "/")
    rows = []
    for crypto in cryptocurrencies:
        configuration_used = os.listdir(name_folder_experiment + "/" + name_folder_result_experiment + "/" + crypto + "/")
        configuration_used.sort(reverse=True)
        # for each configuration:
        for conf in configuration_used:
          cols=["symbol","date","observed_norm","predicted_norm"]
          predictions_csv = pd.read_csv(name_folder_experiment + "/" + name_folder_result_experiment + "/"+str(crypto)+"/"+conf+"/stats/predictions.csv", usecols = cols)
          predictions_csv.columns=["cryptostock_name","date","real_value","predicted_value"]
          #predictions_csv=predictions_csv.dropna(axis=1)
          length= int(len(predictions_csv['cryptostock_name']))
          model=["MultiTarget_Data" for x in range(length)]

          conf_parsed = conf.split("_")
          neurons=[str(conf_parsed[1]) for x in range(length)]
          days=[str(conf_parsed[3]) for x in range(length)]

          predictions_csv.insert(2,"model",model,True)
          predictions_csv.insert(3, "neurons", neurons, True)
          predictions_csv.insert(4, "days", days, True)

          for row in predictions_csv.values:
            rows.append(row)

    cols_n = ["cryptostock_name","date","model","neurons","days","real_value","predicted_value"]
    final_csv= pd.DataFrame(rows,columns=cols_n)
    final_csv.to_csv(name_folder_experiment + "/" + name_folder_result_experiment + "/joined_predictions.csv")

    return



def plot_graphs(input_data, list_crypto, list_model, list_neurons, list_days, output_path):
    data = pd.read_csv(input_data)
    for name, neurons, days in product(list_crypto, list_neurons, list_days):
        data_cutted = data[(data["cryptostock_name"] == name) & (data["neurons"] == neurons) & (data["days"] == days)]
        fig = plt.figure(figsize=(12, 7))
        ax = fig.add_subplot(111)
        # max_values = []
        labels = []
        for (m, i) in zip(list_model, range(0, len(list_model), 1)):
            data_cutted_model_oriented = data_cutted[data["model"] == m]
            if (i == 0):
                ax.plot(range(0, len(data_cutted_model_oriented["date"]), 1),
                        data_cutted_model_oriented["real_value"])
                # max_values.append(max(data_cutted_model_oriented["real_value"]))
                labels.append("REAL")
            ax.plot(range(0, len(data_cutted_model_oriented["date"]), 1),
                    data_cutted_model_oriented["predicted_value"])
            # max_values.append(max(data_cutted_model_oriented["predicted_value"]))
            labels.append("PREDICTED_" + str(m))
        # max_abs = max(max_values)
        plt.title(str(name) + " - #Neurons:" + str(neurons) + " - Previous days:" + str(days))
        plt.xticks(numpy.arange(12), data_cutted_model_oriented["date"], rotation=65)
        # plt.yticks(numpy.arange(0.0, max_abs, (max_abs / 100 * 5)))
        plt.ylabel('Value')
        plt.legend(labels, loc=4)
        plt.grid()
        fig.tight_layout()
        name_fig = str(name) + "_" + str(neurons) + "_" + str(days)
        fig.savefig(output_path + name_fig + ".png")
    return


def generate_linechart_png(path,folder,temporal_sequence,neuronsLSTM,cluster):
    join_predictions(path, "Result")
    cryptocurrenciesSymbol=[]
    for id in cluster:
        for crypto in cluster[id]:
            cryptocurrenciesSymbol.append(crypto)

    folder_creator(path+"/Report/linechartimages",1)
    os.makedirs(path+"/Report/linechartimages/"+folder+"/", exist_ok=True)
    plot_graphs(path+"/Result/joined_predictions.csv", cryptocurrenciesSymbol,["MultiTarget_Data"], neuronsLSTM, temporal_sequence,
                path+"/Report/linechartimages/"+ folder+"/")


