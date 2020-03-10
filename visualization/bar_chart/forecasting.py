import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from itertools import product
import os
from utility.folder_creator import folder_creator
from pathlib import Path

#for forecasting
def report_configurations(temporal_sequence, num_neurons, experiment_folder,
                                       results_folder, report_folder, output_filename):
    # Folder creator
    experiment_and_report_folder = experiment_folder + report_folder + "/"
    experiment_and_result_folder = experiment_folder + results_folder + "/"
    folder_creator(experiment_and_report_folder, 1)

    #if exp_type=="single_target":
    kind_of_report = "configurations_oriented"
    folder_creator(experiment_and_report_folder + kind_of_report + "/", 1)

    #read cryptocurrencies
    cryptocurrencies = os.listdir(experiment_and_result_folder)

    #create dictionary for overall output
    overall_report = {'model': [], 'mean_rmse_norm': [], 'mean_rmse_denorm': []}

    for window, num_neurons in product(temporal_sequence,num_neurons):
        configuration = "LSTM_{}_neurons_{}_days".format(num_neurons,window)

        model_report = {'crypto_symbol': [], 'rmse_list_norm': [], 'rmse_list_denorm': []}

        for crypto in cryptocurrencies:
            #read the error files of a specific crypto
            errors_file = pd.read_csv(experiment_and_result_folder + crypto + "/" + configuration + "/stats/errors.csv",
                index_col=0, sep=',')

            #populate the dictionary
            model_report['crypto_symbol'].append(crypto)
            model_report['rmse_list_norm'].append(errors_file["rmse_norm"])
            model_report['rmse_list_denorm'].append(errors_file["rmse_denorm"])

        #Folder creator
        folder_creator(experiment_and_report_folder + kind_of_report + "/" + configuration + "/",0)

        average_rmse_normalized = np.mean(model_report['rmse_list_norm'])
        average_rmse_denormalized = np.mean(model_report['rmse_list_denorm'])
        configuration_report = {"Average_RMSE_norm": [], "Average_RMSE_denorm": []}
        configuration_report["Average_RMSE_norm"].append(average_rmse_normalized)
        configuration_report["Average_RMSE_denorm"].append(average_rmse_denormalized)

        pd.DataFrame(configuration_report).to_csv(
            experiment_and_report_folder + kind_of_report + "/" + configuration + "/report.csv",index=False)

        #populate overall report
        overall_report['model'].append(configuration)
        overall_report['mean_rmse_norm'].append(average_rmse_normalized)
        overall_report['mean_rmse_denorm'].append(average_rmse_denormalized)

    #overall report to dataframe
    pd.DataFrame(overall_report).to_csv(
        experiment_and_report_folder + kind_of_report + "/" + output_filename + ".csv",index=False)

    #plot overall report
    plot_report(
        path_file= experiment_and_report_folder+ kind_of_report + "/" + output_filename + ".csv",
        x_data="model", column_of_data="mean_rmse_norm", label_for_values_column="RMSE (Average)",
        label_x="Configurations", title_img="Average RMSE - Configurations Oriented",
        destination= experiment_and_report_folder + kind_of_report + "/",
        name_file_output="bargraph_RMSE_configurations_oriented")

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
        report_dic = {'configuration': [], 'RMSE_normalized': []}
        #get the configurations used by the name of their folder
        configurations = os.listdir(experiment_and_result_folder + crypto + "/")
        configurations.sort(reverse=True)

        # for each configuration
        for configuration in configurations:
            # save the configuration's name in the dictionary
            report_dic['configuration'].append(configuration)

            #read 'predictions.csv' file
            errors_file = pd.read_csv(
               experiment_and_result_folder + crypto + "/" + configuration + "/stats/errors.csv")

            #get the mean of the rmse (normalized)
            avg_rmse_norm = errors_file["rmse_norm"].mean()
            #save in the dictionary
            report_dic['RMSE_normalized'].append(float(avg_rmse_norm))

            # get the mean of the rmse (denormalized)
            #avg_rmse_denorm = errors_file["rmse_denorm"].mean()
            # save in the dictionary
            #report_dic['RMSE_denormalized'].append(float(avg_rmse_denorm))

        # save as '.csv' the dictionary in CRYPTO_FOLDER_PATH
        pd.DataFrame(report_dic).to_csv(
            experiment_and_report_folder + kind_of_report + "/" + crypto + "/" + output_filename + ".csv",index=False)

        plot_report(
            path_file=experiment_and_report_folder + kind_of_report + "/" + crypto + "/" + output_filename + ".csv",
            x_data="configuration", column_of_data="RMSE_normalized", label_for_values_column="RMSE (Average)",
            label_x="Configurations", title_img="Average RMSE - " + str(crypto),
            destination=experiment_and_report_folder + kind_of_report + "/" + crypto + "/",
            name_file_output="bargraph_RMSE_" + str(crypto))
    return

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
    f.savefig(destination+name_file_output, bbox_inches='tight', pad_inches=0)
    return

