import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from utility.folder_creator import folder_creator


def report_configurations(temporal_sequence, num_neurons, experiment_folder,
                                       results_folder, report_folder, name_output_files):
    kind_of_report = "configurations_oriented"

    folder_creator(experiment_folder + "/" + report_folder + "/",1)
    folder_creator(experiment_folder + "/" + report_folder + "/" + kind_of_report + "/", 1)

    cryptocurrencies = os.listdir(experiment_folder + "/" + results_folder + "/")

    overall_report = {'model': [], 'mean_rmse_norm': [], 'mean_rmse_denorm': []}

    for window, num_neurons in product(temporal_sequence,num_neurons):
        configuration = "LSTM_{}_neurons_{}_days".format(num_neurons,window)
        model_report = {'crypto_symbol': [], 'rmse_list_norm': [], 'rmse_list_denorm': []}

        for crypto in cryptocurrencies:
            errors_file = pd.read_csv(
                experiment_folder + "/" + results_folder + "/" + crypto + "/" + configuration + "/stats/errors.csv",
                index_col=0, sep=',')

            model_report['crypto_symbol'].append(crypto)
            model_report['rmse_list_norm'].append(errors_file["rmse_norm"])
            model_report['rmse_list_denorm'].append(errors_file["rmse_denorm"])

        folder_creator(experiment_folder+"/"+results_folder+ "/" + kind_of_report + "/" + configuration + "/",0)

        average_rmse_normalized = np.mean(model_report['rmse_list_norm'])
        average_rmse_denormalized = np.mean(model_report['rmse_list_denorm'])

        configuration_report = {"Average_RMSE_norm": [], "Average_RMSE_denorm": []}
        configuration_report["Average_RMSE_norm"].append(average_rmse_normalized)
        configuration_report["Average_RMSE_denorm"].append(average_rmse_denormalized)

        pd.DataFrame(configuration_report).to_csv(
            experiment_folder + "/" + report_folder + "/" + kind_of_report + "/" + configuration + "/report.csv")

        overall_report['model'].append(configuration)
        overall_report['mean_rmse_norm'].append(average_rmse_normalized)
        overall_report['mean_rmse_denorm'].append(average_rmse_denormalized)

    pd.DataFrame(overall_report).to_csv(
        name_folder_experiment + "/" + name_folder_report + "/" + kind_of_report + "/" + name_output_files + ".csv")
    plot_report(
        path_file=name_folder_experiment + "/" + name_folder_report + "/" + kind_of_report + "/" + name_output_files + ".csv",
        x_data="model", column_of_data="mean_rmse_norm", label_for_values_column="RMSE (Average)",
        label_x="Configurations", title_img="Average RMSE - Configurations Oriented",
        destination=name_folder_experiment + "/" + name_folder_report + "/" + kind_of_report + "/",
        name_file_output="bargraph_RMSE_configurations_oriented")
    return


def report_stockseries(name_folder_experiment, name_folder_result_experiment, name_folder_report,
                                    name_files_output):
    kind_of_report = "stockseries_oriented"
    os.makedirs(name_folder_experiment + "/" + name_folder_report + "/", exist_ok=True)
    os.makedirs(name_folder_experiment + "/" + name_folder_report + "/" + kind_of_report + "/", exist_ok=True)

    stock_series = os.listdir(name_folder_experiment + "/" + name_folder_result_experiment + "/")
    # for each stock series:
    for s in stock_series:
        STOCK_FOLDER_PATH = name_folder_experiment + "/" + name_folder_report + "/" + kind_of_report + "/" + s + "/"
        os.makedirs(STOCK_FOLDER_PATH, exist_ok=True)
        single_series_report_dict = {'configuration': [], 'RMSE_normalized': [], 'RMSE_denormalized': []}
        configuration_used = os.listdir(name_folder_experiment + "/" + name_folder_result_experiment + "/" + s + "/")
        configuration_used.sort(reverse=True)
        # for each configuration:
        for c in configuration_used:
            # save name of configuration in dictionary
            single_series_report_dict['configuration'].append(c)
            # read 'predictions.csv' file
            errors_file = pd.read_csv(
                name_folder_experiment + "/" + name_folder_result_experiment + "/" + s + "/" + c + "/stats/errors.csv")
            # perform RMSE_norm and save in dictionary
            avg_rmse_norm = errors_file["rmse_norm"].mean()
            single_series_report_dict['RMSE_normalized'].append(float(avg_rmse_norm))
            # print(float(errors_file['rmse_norm']))
            # perform RMSE_denorm and save in dictionary
            avg_rmse_denorm = errors_file["rmse_denorm"].mean()
            single_series_report_dict['RMSE_denormalized'].append(float(avg_rmse_denorm))
        # save as '.csv' the dictionary in STOCK_FOLDER_PATH
        pd.DataFrame(single_series_report_dict).to_csv(
            name_folder_experiment + "/" + name_folder_report + "/" + kind_of_report + "/" + s + "/" + name_files_output + ".csv")
        plot_report(
            path_file=name_folder_experiment + "/" + name_folder_report + "/" + kind_of_report + "/" + s + "/" + name_files_output + ".csv",
            x_data="configuration", column_of_data="RMSE_normalized", label_for_values_column="RMSE (Average)",
            label_x="Configurations", title_img="Average RMSE - " + str(s),
            destination=name_folder_experiment + "/" + name_folder_report + "/" + kind_of_report + "/" + s + "/",
            name_file_output="bargraph_RMSE_" + str(s))
    return


# Da sistemare ----- WORK IN PROGRESS -------
def plot_report(path_file, x_data, column_of_data, label_for_values_column, label_x, title_img, destination,
                name_file_output):
    report_csv = pd.read_csv(path_file, header=0)
    configurations = report_csv[x_data]
    mean_rmse_normalized = report_csv[column_of_data]
    index = np.arange(len(configurations))
    f = plt.figure()
    plt.bar(index, mean_rmse_normalized)
    plt.ylabel(label_for_values_column, fontsize=10)
    plt.xlabel(label_x, fontsize=10)
    # plt.hlines(np.min(mean_rmse_normalized), 0, len(configurations), linestyles='-', colors='red', linewidth=2, label="Min value")
    # plt.hlines(np.max(mean_rmse_normalized), 0, len(configurations), linestyles='-', colors='red', linewidth=2, label=str(round(np.max(mean_rmse_normalized),4)))
    plt.xticks(index, configurations, fontsize=7, rotation=90)
    plt.title(title_img)
    f.savefig(destination + name_file_output, bbox_inches='tight', pad_inches=0)
    # plt.show()
    return

