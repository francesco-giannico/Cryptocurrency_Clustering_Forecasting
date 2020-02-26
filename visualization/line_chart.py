import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
from itertools import product
import numpy as np

# plot train and validation loss across multiple runs
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
def plot_actual_vs_predicted(input_data, list_crypto, list_model, list_neurons, list_days, output_path):
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
        plt.xticks(np.arange(12), data_cutted_model_oriented["date"], rotation=65)
        # plt.yticks(numpy.arange(0.0, max_abs, (max_abs / 100 * 5)))
        plt.ylabel('Value')
        plt.legend(labels, loc=4)
        plt.grid()
        fig.tight_layout()
        name_fig = str(name) + "_" + str(neurons) + "_" + str(days)
        fig.savefig(output_path + name_fig + ".png")
    return


#todo sistemare... work in progress
def generate_line_chart(path,folder,temporal_sequence,neuronsLSTM,cluster):
    join_predictions(path, "Result")
    cryptocurrenciesSymbol=[]
    for id in cluster:
        for crypto in cluster[id]:
            cryptocurrenciesSymbol.append(crypto)

    folder_creator(path+"/Report/linechartimages",1)
    os.makedirs(path+"/Report/linechartimages/"+folder+"/", exist_ok=True)
    plot_actual_vs_predicted(path+"/Result/joined_predictions.csv", cryptocurrenciesSymbol,["MultiTarget_Data"], neuronsLSTM, temporal_sequence,
                path+"/Report/linechartimages/"+ folder+"/")