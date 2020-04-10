import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from utility.folder_creator import folder_creator
import seaborn as sns

def crypto_oriented(path_multitarget,types):
    #creare un dataframe contenente, per ogni cripto
    #data una cripto, apri ogni file cluster_N.csv e cerca la cripto.
    df_final= pd.DataFrame(columns=['symbol','k_1','k_sqrtN','k_sqrtNby2',
                                    'k_sqrtNby4','k_sqrtNdiv2','k_sqrtNdiv4','lowest_single_target',
                                    'baseline'])

    for csv in os.listdir(os.path.join(path_multitarget,"outputs_k1","reports")):
      if csv.endswith(".csv"):
        df=pd.read_csv(os.path.join(path_multitarget,"outputs_k1","reports",csv))
        df_final.symbol=df.symbol
        df_final.k_1=df.avg_rmse_multi
        df_final.lowest_single_target=df.avg_rmse_single
        df_final.baseline = df.baseline
    df_final=df_final.set_index("symbol")

    for k in types:
        if k!="outputs_k1":
            for csv in os.listdir(os.path.join(path_multitarget,k,"reports")):
              if csv.endswith(".csv"):
                df=pd.read_csv(os.path.join(path_multitarget,k,"reports",csv))
                index=k.replace("outputs_","")
                for crypto in df.symbol.values:
                 df = df.set_index("symbol")
                 df_final.at[crypto,index] = df.at[crypto,"avg_rmse_multi"]
                 df=df.reset_index()

    df_final=df_final.reset_index()
    folder_creator(os.path.join(path_multitarget,"reports_multi"),1)

    for crypto in df_final.symbol.values:
        plt.figure(figsize=(20,20))
        ax = sns.barplot(data=df_final[df_final.symbol == crypto], ci=None)
        title = crypto
        plt.title(title)
        ax.set(xlabel='Competitors', ylabel='RMSE')
        plt.savefig(os.path.join(path_multitarget,"reports_multi", crypto + ".png"), dpi=100)


def compare_multi_baseline_single_target_chart(input_path,output_path):

    for cluster in os.listdir(input_path):
        if cluster != "single_crypto" and cluster != "averages":
            folder_creator(os.path.join(output_path,cluster.replace(".csv","")),1)
            df = pd.read_csv(os.path.join(input_path, cluster))
            for crypto in df.symbol.values:
                #print(df[df.symbol==crypto])
                plt.figure()
                df_new= pd.DataFrame(columns=['symbol','lowest_rmse_multi','lowest_rmse_single','baseline'])
                df_new['symbol']=df.symbol
                df_new['lowest_rmse_multi'] = df.avg_rmse_multi
                df_new['lowest_rmse_single'] = df.avg_rmse_single
                df_new['baseline'] = df.baseline
                ax = sns.barplot(data=df_new[df_new.symbol==crypto], ci=None)
                title = crypto
                plt.title(title)
                ax.set(xlabel='Competitors', ylabel='RMSE')
                plt.savefig(os.path.join(output_path,cluster.replace(".csv",""),crypto + ".png"), dpi=100)


def compare_avg_multi_baseline_single_target_chart(input_path,output_path):
    for cluster in os.listdir(input_path):
        if cluster != "single_crypto" and cluster!="averages":
            df=pd.read_csv(os.path.join(input_path,cluster))
            plt.figure()
            ax=sns.barplot(data=df,ci=None)
            title=cluster.replace(".csv","")+" ["
            for symbol in df['symbol'].values:
                title+=symbol +","
            title+="]"
            plt.title(title)
            ax.set(xlabel='Competitors', ylabel='Average RMSE')
            plt.savefig(output_path+"/"+cluster.replace(".csv","")+".png",dpi=100)


def compare_multi_baseline_single_target(path_baseline,path_single,path_multi,output_path):
    folder_creator(output_path, 1)
    folder_creator(os.path.join(output_path,"averages"),1)
    folder_creator(os.path.join(output_path,"single_crypto"), 1)
    for cluster in os.listdir(path_multi):
        output_file = {'symbol': [], 'avg_rmse_multi': [], 'avg_rmse_single': [], 'baseline': []}
        #vai in result del cluster in corso
        for crypto in os.listdir(os.path.join(path_multi,cluster,"result")):
            #leggere da baseline
            file = open(os.path.join(path_baseline,crypto), "r")
            rmse_baseline= float(file.read())
            file.close()

            #lowest single target
            min_single=100
            conf_name_single=""
            for configuration in os.listdir(os.path.join(path_single,crypto)):
                df=pd.read_csv(os.path.join(path_single,crypto,configuration,"stats/errors.csv"),header=0)
                if df["rmse_norm"][0]< min_single:
                    min_single=df["rmse_norm"][0]
                    conf_name_single=configuration

            # lowest multi target
            min_multi = 100
            conf_name_multi= ""
            for configuration in os.listdir(os.path.join(path_multi,cluster,"result",crypto)):
                df = pd.read_csv(os.path.join(path_multi,cluster,"result",crypto,configuration,"stats/errors.csv"), header=0)
                if df["rmse_norm"][0] < min_multi:
                    min_multi= df["rmse_norm"][0]
                    conf_name_multi = configuration

            output_file['symbol'].append(crypto)
            output_file['avg_rmse_single'].append(min_single)
            output_file['avg_rmse_multi'].append(min_multi)
            output_file['baseline'].append(rmse_baseline)

            pd.DataFrame(data=output_file).to_csv(os.path.join(output_path,cluster+".csv"), index=False)
    compare_avg_multi_baseline_single_target_chart(output_path,os.path.join(output_path,"averages"))
    compare_multi_baseline_single_target_chart(output_path,os.path.join(output_path,"single_crypto"))


def report_clustering(path, pathToSave):
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
    ax.set_xticklabels(ks_x, rotation=90, ha="right")

    # ax.legend((rects1[0],rects2[0]), (exp_name[0],exp_name[1]))
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.grid(linewidth=0.2, color='black')
    plt.savefig(pathToSave + "/cluster.png")
    # plt.show()"""


def results_by_cluster(path, algorithms):
    for algorithm_k in algorithms:
        try:
            clusters = get_clusters(algorithm_k)
            N = len(clusters)
        except:
            N = 1

        df = pd.read_csv(path + "/" + algorithm_k + "/myresults.csv",
                         usecols=["cluster_id", "crypto_name", "average_rmse_norm"])
        dfOriginal = df
        for clusterID in range(N):
            partial = df[df['cluster_id'] == clusterID]
            """avg = np.average(partial['average_rmse_norm'])
            partial = partial[partial['cluster_id'] ]"""
            partial = partial.sort_values(by=['average_rmse_norm'], ascending=False)
            pathFrom = path + "/" + algorithm_k + "/cluster_" + str(clusterID) + "/ordered_list_by_rmse.csv"
            partial.to_csv(pathFrom)
            report_clustering(pathFrom, path + "/" + algorithm_k + "/cluster_" + str(clusterID))

        try:
            partial1 = dfOriginal.sort_values(by=['average_rmse_norm'], ascending=False)
            partial1.to_csv(path + "/" + algorithm_k + "/ordered_list_by_rmse_all.csv")
            pathFrom = path + "/" + algorithm_k + "/ordered_list_by_rmse_all.csv"
            report_clustering(pathFrom, path + "/" + algorithm_k)
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


def barchartComparison(experiments):
    pathGeneric = "crypto_clustering/experiment_common/"
    name = "/myresults.csv"
    csvs = []

    for folder in os.listdir(pathGeneric):
        if (folder != "clusters" and folder != "cutData" and folder != "horizontalDataset"):
            csvs.append(
                pd.read_csv(pathGeneric + folder + name, usecols=['cluster_id', 'crypto_name', 'average_rmse_norm']))

    for experiment in experiments:
        if experiment != "experiment_common":
            pathGeneric = "crypto_clustering/" + experiment + "/"
            folder_creator(pathGeneric + "reports", 1)
            for folder in os.listdir(pathGeneric):
                name = "/myresults.csv"
                if (
                        folder != "clusters" and folder != "cutData" and folder != "horizontalDataset" and folder != "reports"):
                    csvs.append(pd.read_csv(pathGeneric + folder + name,
                                            usecols=['cluster_id', 'crypto_name', 'average_rmse_norm']))
            ks_x = []
            i = 0
            while i < len(csvs[0]["crypto_name"]):
                ks_x.append(str(csvs[0]["crypto_name"][i]))
                i += 1

            ind = np.arange(len(ks_x))  # the x locations for the groups
            ind = ind * 1.5
            width = 0.15  # the width of the bars

            fig = plt.figure(figsize=(30, 10), dpi=150)
            ax = fig.add_subplot(111)
            ax.set_title("Comparision: Single target VS Multitarget (NO INDICATORS)")

            csv1V = []
            csv2V = []
            csv3V = []
            csv4V = []
            csv5V = []
            csv6V = []
            csv7V = []
            for name in ks_x:
                crypto1 = csvs[0][csvs[0]["crypto_name"] == name]
                crypto2 = csvs[1][csvs[1]["crypto_name"] == name]
                crypto3 = csvs[2][csvs[2]["crypto_name"] == name]
                crypto4 = csvs[3][csvs[3]["crypto_name"] == name]
                crypto5 = csvs[4][csvs[4]["crypto_name"] == name]
                crypto6 = csvs[5][csvs[5]["crypto_name"] == name]
                crypto7 = csvs[6][csvs[6]["crypto_name"] == name]
                csv1V.append(crypto1["average_rmse_norm"].values[0])
                csv2V.append(crypto2["average_rmse_norm"].values[0])
                csv3V.append(crypto3["average_rmse_norm"].values[0])
                csv4V.append(crypto4["average_rmse_norm"].values[0])
                csv5V.append(crypto5["average_rmse_norm"].values[0])
                csv6V.append(crypto6["average_rmse_norm"].values[0])
                csv7V.append(crypto7["average_rmse_norm"].values[0])

            rects1 = ax.bar(ind, csv1V, width, color='green')
            rects2 = ax.bar(ind + width, csv2V, width, color='orange')
            space = width * 2
            rects3 = ax.bar(ind + space, csv3V, width, color='brown')
            space = width * 3
            rects4 = ax.bar(ind + space, csv4V, width, color='blue')
            space = width * 4
            rects5 = ax.bar(ind + space, csv5V, width, color='cyan')
            space = width * 5
            rects6 = ax.bar(ind + space, csv6V, width, color='olive')
            space = width * 6
            rects7 = ax.bar(ind + space, csv7V, width, color='magenta')

            ax.set_ylabel('Average (RMSE)')
            ax.set_xlabel('Crypto names')
            ax.set_xticks(ind)
            ax.set_xticklabels(ks_x, rotation=45, ha="right")
            # ax.legend((rects1[0], rects2[0]), (algorithms[0], algorithms[1]))
            # ax.legend((rects1[0], rects2[0], rects3[0]),(algorithms[0], algorithms[1], algorithms[2]))
            # ax.legend((rects1[0],rects2[0],rects3[0],rects4[0]), (algorithms[0],algorithms[1],algorithms[2],algorithms[3]))

            # ax.legend((rects1[0],rects2[0],rects3[0],rects4[0], rects5[0]), (algorithms[0],algorithms[1],algorithms[2],algorithms[3],algorithms[4]))

            ax.legend((rects1[0], rects2[0], rects3[0], rects4[0], rects5[0], rects6[0], rects7[0]),
                      ("multitarget", "singletarget", "sqrtN", "sqrtNby2", "sqrtNby4", "sqrtNdiv2", "sqrtNdiv4"))
            ax.set_axisbelow(True)
            plt.tight_layout()
            plt.grid(linewidth=0.3, color='black')
            plt.savefig(pathGeneric + "reports/comparison_singleVSmulti.png")
            # plt.show()


def overall_report(title, filenameout, filenamein):
    pathGeneric = "crypto_clustering/reports"
    name = "/" + filenamein
    csv = pd.read_csv(pathGeneric + "/" + name + ".csv")
    ks_x = []

    exp_name = ["dtw", "pearson", "multitarget", "singletarget"]

    yvals = []
    zvals = []
    gvals = []
    mvals = []
    ks = ['sqrtN', "sqrtNdiv2", "sqrtNby2", "sqrtNdiv4", "sqrtNby4"]

    tofind = "multitarget"
    filt1 = csv[csv["k_description"] == tofind]
    gvals.append(filt1['average_rmse_norm'].values[0])

    tofind = "singletarget"
    filt1 = csv[csv["k_description"] == tofind]
    mvals.append(filt1['average_rmse_norm'].values[0])

    for k in ks:
        tofind = "dtw_" + k
        tofind2 = "pearson_" + k
        filt1 = csv[csv["k_description"] == tofind]
        filt2 = csv[csv["k_description"] == tofind2]
        yvals.append(filt1['average_rmse_norm'].values[0])
        zvals.append(filt2['average_rmse_norm'].values[0])
        # print(filt1['k'].values[0])
        ks_x.append(k + "=" + str(filt1['k'].values[0]))

    ind = np.arange(len(ks_x))  # the x locations for the groups
    ind = ind * 2.3
    width = 0.30  # the width of the bars

    fig = plt.figure(figsize=(25, 10), dpi=70)
    ax = fig.add_subplot(111)
    ax.set_title(title)

    rects1 = ax.bar(ind, yvals, width, color='green')
    rects2 = ax.bar(ind + width, zvals, width, color='orange')
    rects3 = ax.bar(ind + width + width, gvals, width, color='blue')
    rects4 = ax.bar(ind + width + width + width, mvals, width, color='olive')

    def autolabel(rects):
        # Attach a text label above each bar displaying its height

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

    ax.legend((rects1[0], rects2[0], rects3[0], rects4[0]), (exp_name[0], exp_name[1], exp_name[2], exp_name[3]))
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.grid(linewidth=0.2, color='black')
    plt.savefig(pathGeneric + "/" + filenameout + ".png")
    # plt.show()"""
