import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
from itertools import product
import numpy as np

# plot train and validation loss across multiple runs
from utility.clustering_utils import merge_predictions
from utility.folder_creator import folder_creator
from utility.reader import get_crypto_symbols_from_folder


def plot_train_and_validation_loss(train,test,output_folder):
    plt.plot(train, color='blue', label='Train')
    plt.plot(test, color='orange', label='Validation')
    plt.title('model train vs validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    legend_elements = [Line2D([0], [0], color='blue', lw=2, label='Train'),
                       Line2D([0], [0], color='orange', lw=2, label='Validation'),]
    plt.legend(handles=legend_elements, loc='upper left')
    plt.savefig(output_folder+"model_train_validation_loss.png",dpi=150)


#plot the actual value and the predicted value
def plot_actual_vs_predicted(
        input_data,cryptocurrencies, crypto, list_neurons, list_temporal_sequences, output_path):

    #reads the csv
    data = pd.read_csv(input_data)

    #for crypto, neurons, days in product(cryptocurrencies, list_neurons, list_temporal_sequences):
    #read a specific line from the file
    data_cut = data[(data["symbol"] == crypto)]
    for neurons, days in product(list_neurons, list_temporal_sequences):
        data_cut=data_cut[(data_cut["neurons"] == neurons) & (data["days"] == days)]

        #create a figure
        fig = plt.figure(figsize=(12, 7),dpi=150)

        #create a subplot
        ax=fig.add_subplot(1,1,1)
        plt.title(str(crypto) + " - #Neurons:" + str(neurons) + " - Previous days:" + str(days))
        plt.ylabel('Value')

        labels = []
        #model oriented information
        #data_cut_model_oriented = data_cut[data["model"] == model]
        ax.plot(range(0, len(data_cut["date"]), 1),data_cut["observed_value"])
        labels.append("REAL")
        ax.plot(range(0, len(data_cut["date"]), 1), data_cut["predicted_value"])
        labels.append("PREDICTED")

        plt.xticks(np.arange(12), data_cut["date"], rotation=65)
        plt.legend(labels, loc=4)
        plt.grid()
        fig.tight_layout()
        name_fig = str(crypto) + "_" + str(neurons) + "_" + str(days)
        fig.savefig(output_path + name_fig + ".png")
    return


#todo sistemare... work in progress
def generate_line_chart(experiment_folder,list_temporal_sequences,list_neurons):
    cryptocurrencies = get_crypto_symbols_from_folder(experiment_folder + "result/")

    merge_predictions(experiment_folder, "result")

    #create the folder which will contain the line chart
    for crypto in cryptocurrencies:
        folder_creator(experiment_folder+"/report/line_chart_images/"+crypto,1)
        plot_actual_vs_predicted(experiment_folder+"/result/merged_predictions.csv",
                                 cryptocurrencies,
                                 crypto,
                                 list_neurons,
                                 list_temporal_sequences,
                                 experiment_folder+"/report/line_chart_images/"+ crypto+"/")