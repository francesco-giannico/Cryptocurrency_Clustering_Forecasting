import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from itertools import product
import os
from utility.folder_creator import folder_creator
from pathlib import Path
import seaborn as sns


def overall_macro_avg_recall_baseline(input_file_path,output_path):
    df = pd.read_csv(input_file_path, header=0)
    avg=np.average(df.value)
    df=pd.DataFrame(columns=['average'])
    df=df.append({'average':avg},ignore_index=True)
    plt.figure()
    flatui = ["#3498db"]
    palettes = sns.color_palette(flatui)
    ax = sns.barplot(data=df, ci=None, palette=palettes)
    title = "Baseline overall"
    plt.title(title)
    ax.set(xlabel='All cryptocurrencies', ylabel='Average macro average recall')
    plt.savefig(output_path + "/overall_" + title + ".png", dpi=100)
    plt.close()


def comparison_macro_avg_recall_baseline(input_file_path,output_path):
    df=pd.read_csv(input_file_path,header=0)
    df.set_index('crypto', inplace=True)
    df_t=df.T
    plt.figure()


    flatui = ["#3498db"]
    palettes=sns.color_palette(flatui)
    sns.set(font_scale=0.65, style="white")
    ax = sns.barplot(data=df_t, ci=None,palette=palettes)

    """for bar in ax.patches:
        if bar.get_height() > 6:
            bar.set_color('red')
        else:
            bar.set_color('grey')"""
    title = "Baseline"
    plt.title(title)
    ax.set(xlabel='Cryptocurrencies', ylabel='Macro average recall')
    plt.savefig(output_path+ "/comparison_"+title + ".png", dpi=100)
    plt.close()

#input path single: path to results folder
#input path_baseline: performances
#output_path: report for each crypto
def comparison_macro_avg_recall_single_vs_baseline(input_path_single,input_path_baseline,output_path):
    folder_creator(output_path,0)
    for crypto in os.listdir(input_path_single):

        folder_creator(os.path.join(output_path,crypto),1)
        # read baseline
        file = open(input_path_baseline + crypto+"_macro_avg_recall.txt", "r")
        macro_avg_recall_baseline = file.read()
        file.close()

        #find best configuration
        max_macro_avg_recall = -1
        config = ""
        for configuration in os.listdir(os.path.join(input_path_single, crypto)):
            df = pd.read_csv(os.path.join(input_path_single, crypto, configuration, "stats/macro_avg_recall.csv"), header=0)
            if df["macro_avg_recall"][0] > max_macro_avg_recall:
                max_macro_avg_recall = df["macro_avg_recall"][0]
                config = configuration

        #generate csv containing these info
        df_report=pd.DataFrame()
        df_report = df_report.append(
            {'crypto': crypto, 'model type': 'single_target', 'macro_avg_recall': float(max_macro_avg_recall),'config':config},
            ignore_index=True)
        df_report=df_report.append({'crypto':crypto,'model type':'baseline','macro_avg_recall': float(macro_avg_recall_baseline),'config':'standard'},ignore_index=True)
        df_report.to_csv(os.path.join(output_path,crypto,crypto+"_report.csv"),index=False)
        # generate individual chart
        comparison_macro_avg_recall_single_vs_baseline_plot(df_report,os.path.join(output_path,crypto))

def comparison_macro_avg_recall_single_vs_baseline_plot(df,output_path):
    title = "Single target VS Baseline"
    flatui = ["#3498db", "#FF9633"]
    palettes = sns.color_palette(flatui)
    #sns.set(font_scale=0.65, style='white')
    ax = sns.barplot(x="crypto", y="macro_avg_recall", hue="model type", data=df, palette=palettes)
    plt.title(title)
    ax.set(xlabel='Models', ylabel='Macro average recall')
    plt.savefig(output_path + "/" + title + ".png", dpi=100)
    plt.close()

def overall_macro_avg_recall_single(input_path_single,output_path):
    avg_bests = []
    for crypto in os.listdir(input_path_single):
        # find best configuration
        max_macro_avg_recall = -1
        config = ""
        for configuration in os.listdir(os.path.join(input_path_single, crypto)):
            df = pd.read_csv(os.path.join(input_path_single, crypto, configuration, "stats/macro_avg_recall.csv"),
                             header=0)
            if df["macro_avg_recall"][0] > max_macro_avg_recall:
                max_macro_avg_recall = df["macro_avg_recall"][0]
                config = configuration
        avg_bests.append(max_macro_avg_recall)

    avg_single_target = np.average(avg_bests)
    df_report = pd.DataFrame()
    df_report = df_report.append(
        {'crypto': "Models", 'model type': 'single_target', 'macro_avg_recall': float(avg_single_target)},
        ignore_index=True)
    df_report.to_csv(os.path.join(output_path, "single_average_report.csv"), index=False)
    overall_macro_avg_recall_single_plot(df_report, output_path)
def overall_macro_avg_recall_single_plot(df,output_path):
    title = "Single target"
    flatui = ["#3498db", "#FF9633"]
    palettes = sns.color_palette(flatui)
    ax = sns.barplot(x="crypto", y="macro_avg_recall", hue="model type", data=df, palette=palettes)
    plt.title(title)
    ax.set(xlabel=" ", ylabel='Average macro average recall')
    plt.savefig(output_path + "/" + title + ".png", dpi=100)
    plt.close()

def overall_comparison_macro_avg_recall_simple_vs_baseline(input_path_single,input_path_baseline,output_path):
    avg_bests = []
    for crypto in os.listdir(input_path_single):

        # find best configuration
        max_macro_avg_recall = -1
        config = ""
        for configuration in os.listdir(os.path.join(input_path_single, crypto)):
            df = pd.read_csv(os.path.join(input_path_single, crypto, configuration, "stats/macro_avg_recall.csv"),
                             header=0)
            if df["macro_avg_recall"][0] > max_macro_avg_recall:
                max_macro_avg_recall = df["macro_avg_recall"][0]
                config = configuration
        avg_bests.append(max_macro_avg_recall)

    # average baseline
    file = open(input_path_baseline + "average_macro_avg_recall.txt", "r")
    avg_macro_avg_recall_baseline = file.read()
    file.close()
    avg_single_target = np.average(avg_bests)
    df_report = pd.DataFrame()
    df_report = df_report.append(
        {'crypto': "Models", 'model type': 'single_target', 'macro_avg_recall': float(avg_single_target)},
        ignore_index=True)
    df_report = df_report.append(
        {'crypto': "Models", 'model type': 'baseline', 'macro_avg_recall': float(avg_macro_avg_recall_baseline)},
        ignore_index=True)
    df_report.to_csv(os.path.join(output_path, "single_vs_baseline_report.csv"), index=False)
    overall_comparison_macro_avg_recall_simple_vs_baseline_plot(df_report,output_path)

def overall_comparison_macro_avg_recall_simple_vs_baseline_plot(df,output_path):
    title = "Single target VS Baseline (Average)"
    flatui = ["#3498db", "#FF9633"]
    palettes = sns.color_palette(flatui)
    ax = sns.barplot(x="crypto", y="macro_avg_recall", hue="model type", data=df, palette=palettes)
    plt.title(title)
    ax.set(xlabel=" ",ylabel='Average macro average recall')
    plt.savefig(output_path + "/" + title + ".png", dpi=100)
    plt.close()

#for forecasting
def report_configurations(temporal_sequence, num_neurons, experiment_folder,
                                       results_folder, report_folder, output_filename):
    # Folder creator
    experiment_and_report_folder = experiment_folder + report_folder + "/"
    experiment_and_result_folder = experiment_folder + results_folder + "/"
    folder_creator(experiment_and_report_folder, 1)

    kind_of_report = "configurations_oriented"
    folder_creator(experiment_and_report_folder + kind_of_report + "/", 1)

    #read cryptocurrencies
    cryptocurrencies = os.listdir(experiment_and_result_folder)

    #create dictionary for overall output
    #overall_report = {'model': [], 'mean_rmse_norm': [], 'mean_rmse_denorm': []}
    overall_report = {'model': [], 'mean_accuracy': []}
    overall_report['model'].append("simple prediction model")
    OUTPUT_SIMPLE_PREDICTION = "../modelling/techniques/baseline/simple_prediction/output/"
    file = open(OUTPUT_SIMPLE_PREDICTION + "average_accuracy/average_accuracy.txt", "r")
    value1 = file.read()
    file.close()
    overall_report['mean_accuracy'].append(value1)
    for window, num_neurons in product(temporal_sequence,num_neurons):
        configuration = "LSTM_{}_neurons_{}_days".format(num_neurons,window)

        #model_report = {'crypto_symbol': [], 'rmse_list_norm': [], 'rmse_list_denorm': []}
        model_report = {'crypto_symbol': [], 'accuracy_list': []}
        for crypto in cryptocurrencies:
            #read the error files of a specific crypto
            accuracy_file = pd.read_csv(experiment_and_result_folder + crypto + "/" + configuration + "/stats/macro_avg_recall.csv",
                index_col=0, sep=',')

            #populate the dictionary
            model_report['crypto_symbol'].append(crypto)
            model_report['accuracy_list'].append(accuracy_file["macro_avg_recall"])
            #model_report['rmse_list_denorm'].append(errors_file["rmse_denorm"])

        #Folder creator
        folder_creator(experiment_and_report_folder + kind_of_report + "/" + configuration + "/",0)

        average_macro_avg_recall = np.mean(model_report['accuracy_list'])
        #average_rmse_denormalized = np.mean(model_report['rmse_list_denorm'])

        #configuration_report = {"Average_RMSE_norm": [], "Average_RMSE_denorm": []}
        configuration_report = {"Average_macro_avg_recall": []}
        configuration_report["Average_macro_avg_recall"].append(average_macro_avg_recall)
        #configuration_report["Average_RMSE_denorm"].append(average_rmse_denormalized)

        pd.DataFrame(configuration_report).to_csv(
            experiment_and_report_folder + kind_of_report + "/" + configuration + "/report.csv",index=False)

        #populate overall report
        overall_report['model'].append(configuration)
        overall_report['mean_accuracy'].append(average_macro_avg_recall)
        #overall_report['mean_rmse_denorm'].append(average_rmse_denormalized)

    #overall report to dataframe
    pd.DataFrame(overall_report).to_csv(
        experiment_and_report_folder + kind_of_report + "/" + output_filename + ".csv",index=False)

    #plot overall report
    plot_report(
        path_file= experiment_and_report_folder+ kind_of_report + "/" + output_filename + ".csv",
        x_data="model", column_of_data="mean_accuracy", label_for_values_column="Macro avg recall",
        label_x="Configurations", title_img="Macro avg recall - Configurations Oriented",
        destination= experiment_and_report_folder + kind_of_report + "/",
        name_file_output="bargraph_accuracy_configurations_oriented")

    return

#for forecasting
def report_crypto(experiment_folder, result_folder, report_folder,output_filename):
    kind_of_report = "crypto_oriented"

    # Folder creator
    experiment_and_report_folder = experiment_folder + report_folder + "/"
    experiment_and_result_folder = experiment_folder + result_folder + "/"
    folder_creator(experiment_and_report_folder, 0)
    folder_creator(experiment_and_report_folder + kind_of_report + "/", 0)

    # read cryptocurrencies
    cryptocurrencies = os.listdir(experiment_and_result_folder)

    # for each crypto:
    for crypto in cryptocurrencies:
        #folder creator
        CRYPTO_FOLDER_PATH = experiment_and_report_folder + kind_of_report + "/" + crypto + "/"
        folder_creator(CRYPTO_FOLDER_PATH,1)

        #dictionary for report
        #report_dic = {'configuration': [], 'RMSE_normalized': [], 'RMSE_denormalized': []}
        report_dic = {'configuration': [], 'Accuracy': []}
        #get the configurations used by the name of their folder
        configurations = os.listdir(experiment_and_result_folder + crypto + "/")
        configurations.sort(reverse=True)


        # for each configuration
        for configuration in configurations:
            # save the configuration's name in the dictionary
            report_dic['configuration'].append(configuration)

            #read 'predictions.csv' file
            macro_avg_recall_file = pd.read_csv(
               experiment_and_result_folder + crypto + "/" + configuration + "/stats/macro_avg_recall.csv")

            #get the mean of the rmse (normalized)
            avg_accuracy = macro_avg_recall_file["macro_avg_recall"].mean()
            #save in the dictionary
            report_dic['Accuracy'].append(float(avg_accuracy))

            # get the mean of the rmse (denormalized)
            #avg_rmse_denorm = errors_file["rmse_denorm"].mean()
            # save in the dictionary
            #report_dic['RMSE_denormalized'].append(float(avg_rmse_denorm))

        # save as '.csv' the dictionary in CRYPTO_FOLDER_PATH
        pd.DataFrame(report_dic).to_csv(
            experiment_and_report_folder + kind_of_report + "/" + crypto + "/" + output_filename + ".csv",index=False)

        plot_report(
            path_file=experiment_and_report_folder + kind_of_report + "/" + crypto + "/" + output_filename + ".csv",
            x_data="configuration", column_of_data="Accuracy", label_for_values_column="Macro avg recall (average)",
            label_x="Configurations", title_img="Average Macro avg recall - " + str(crypto),
            destination=experiment_and_report_folder + kind_of_report + "/" + crypto + "/",
            name_file_output="bargraph_macro_avg_recall_" + str(crypto))
    return

"""def report_single_vs_simple(input_path,cryptocurrencies):
    

    #reads the csv (merged_predictions.csv)
    df = pd.read_csv(input_path,sep=",")
    

    #for crypto, neurons, days in product(cryptocurrencies, list_neurons, list_temporal_sequences):
    #read a specific line from the file
    data_cut = data[(data["symbol"] == crypto)]
    for neurons, days in product(list_neurons, list_temporal_sequences):
        data_cut=data_cut[(data_cut["neurons"] == neurons) & (data["days"] == days)]"""

#for forecasting
def plot_report(path_file, x_data, column_of_data, label_for_values_column, label_x, title_img, destination,
                name_file_output):

    #read the report
    report_csv = pd.read_csv(path_file, header=0)

    #read some columns
    configurations = report_csv[x_data]
    mean_rmse_normalized = report_csv[column_of_data]

    #define the index
    index = np.arange(len(configurations))

    #create figure
    f = plt.figure()

    #bar chart
    plt.bar(index, mean_rmse_normalized)

    #set the labels
    plt.ylabel(label_for_values_column, fontsize=10)
    plt.xlabel(label_x, fontsize=10)
    plt.title(title_img)

    #customize the labels by rotating of 90 degree, for example
    plt.xticks(index, configurations, fontsize=7, rotation=90)

    #serialization
    f.savefig(destination+name_file_output, bbox_inches='tight', pad_inches=0,dpi=120)
    return

