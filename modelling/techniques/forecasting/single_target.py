import os
from itertools import product
import numpy as np
import pandas as pd

from modelling.techniques.forecasting.testing.test_set import generate_testset
from modelling.techniques.forecasting.training.training import prepare_input_forecasting, fromtemporal_totensor,get_training_testing_set
from utility.folder_creator import folder_creator

np.random.seed(0)

# TENSOR_PATH = "../crypto_TensorData"
# Parameters of experiments
# temporal_sequence_considered = [30, 100, 200]
# number_neurons_LSTM = [128, 256]
# features_to_exclude_from_scaling = ['Symbol']

PREPROCESSED_PATH="../preparation/preprocessed_dataset/cleaned/final/"
def single_target1(EXPERIMENT_PATH, DATA_PATH, TENSOR_DATA_PATH, window_sequence, number_neurons, learning_rate,
                   testing_set):

    #################### FOLDER SETUP ####################
    MODELS_PATH = "models"
    RESULT_PATH = "result"
    REPORT_FOLDER_NAME = "report"

    for crypto in os.listdir(DATA_PATH):
        crypto_name = crypto.replace(".csv", "")

        # create a folder for data in tensor format
        folder_creator(TENSOR_DATA_PATH + "/" + crypto_name,0)
        # create a folder for results
        folder_creator(EXPERIMENT_PATH + "/" + MODELS_PATH + "/" + crypto_name, 1)
        folder_creator(EXPERIMENT_PATH + "/" + RESULT_PATH + "/" + crypto_name, 1)

        dataset, features, features_without_date, df1=  \
            prepare_input_forecasting(PREPROCESSED_PATH,DATA_PATH,crypto)
        """print(dataset['Close'].head())
        print(scaler_target_feature['Close'].head())"""
        #print(list(product(temporal_sequence, number_neurons)))
        #[(30, 128), (30, 256), (100, 128), (100, 256), (200, 128), (200, 256)]
        #print(np.array(dataset)[0])#prendo l a prima riga del dataset (2018-... bla bla bal)
        for window, neurons in product(window_sequence, number_neurons):
            print("Crypto_symbol", "\t", "Window_sequence", "\t", "Neurons")
            print(crypto, "\t","\t", window, "\t","\t", neurons)
            #non prende in input i neuroni questo.
            dataset_tensor_format = fromtemporal_totensor(np.array(dataset), window,
                                                   TENSOR_DATA_PATH + "/" + crypto_name + "/",
                                                   crypto_name)
            # DICTIONARY FOR STATISTICS
            predictions_file = {'symbol': [], 'date': [], 'observed_norm': [], 'predicted_norm': [],
                                'observed_denorm': [],'predicted_denorm': []}
            errors_file = {'symbol': [], 'rmse_norm': [], 'rmse_denorm': []}

            # define a name for this configuration
            configuration_name = "LSTM_" + str(neurons) + "_neurons_" + str(window) + "_days"
            # Create a folder to save
            # - best model checkpoint
            # - statistics (results)
            best_model = "model"
            statistics = "stats"
            folder_creator(EXPERIMENT_PATH + "/" + MODELS_PATH + "/" + crypto_name + "/" + configuration_name + "/" + best_model,1)
            folder_creator(EXPERIMENT_PATH + "/" + RESULT_PATH + "/" + crypto_name + "/" + configuration_name + "/" + statistics,1)

            #starting from the testing set
            for date_to_predict in testing_set:
                #todo-balla, in realtà addestro fino a 2 giorni prima della data da predire.
                print("Training till: ", pd.to_datetime(date_to_predict))
                """the format of train and test is the following one:
                 [
                    [[items],[items]],
                    [[items],[items]],
                    ....
                    [[items],[items]],
                 ]
                thus for element accessing there are the following three indexes:
                  1)e.g [[items],[items]]
                  2)e.g [items],[items]
                  3)e.g items
                """
                train, test = get_training_testing_set(dataset_tensor_format, date_to_predict)
                #todo il test set generato forse non ha senso che sia cosi. Rivedilo.

                # ['2018-01-01' other numbers separated by comma],it removes the date.
                train = train[:, :, 1:]
                test = test[:, :, 1:]

                index_of_feature_close=features_without_date.index('Close')
                #remove the last day before the day to predict:
                #e.g date to predict 2019-01-07 thus the data about 2019-01-06 will be discarded.
                #e.g [[items],[items2],[items3]] becames [[items1],[items2]]
                x_train= train[:, :-1, :]
                # remove the last day before the day to predict, by doing -1
                # returns an array with all the values of the feature close
                y_train =train[:, -1,  index_of_feature_close]

                # NOTE: in the testing set we must have the dates to evaluate the experiment without the date to forecast!!!
                # remove the day to predict
                # e.g date to predict 2019-01-07 thus the data about 2019-01-07 will be discarded.
                # e.g [[items],[items2],[items3]] becames [[items1],[items2]]
                x_test = test[:, :-1, :]
                # remove the last day before the day to predict, by doing -1
                # returns an array with all the values of the feature close to predict!
                y_test = test[:, -1, index_of_feature_close]

                break
                # Does the training
                """if data_tester == testing_set[0]:
                    model, history = experiments.train_model(x_train, y_train, x_test, y_test, lstm_neurons=neurons,
                                                             learning_rate=learning_rate,
                                                             dropout=0.2,
                                                             epochs=100,
                                                             batch_size=256,
                                                             dimension_last_layer=1,
                                                             model_path=EXPERIMENT + "/" + MODELS_PATH + "/" + stock_name + "/" + configuration_name + "/" + best_model + "/")
                else:
                    model, history = experiments.train_model(x_train, y_train, x_test, y_test, lstm_neurons=neurons,
                                                             learning_rate=learning_rate,
                                                             dropout=0.2,
                                                             epochs=100,
                                                             batch_size=256, dimension_last_layer=1, model=model,
                                                             model_path=EXPERIMENT + "/" + MODELS_PATH + "/" + stock_name + "/" + configuration_name + "/" + best_model + "/")
"""
            break
        break

def single_target(EXPERIMENT_PATH, DATA_PATH, TENSOR_DATA_PATH, temporal_sequence, number_neurons, learning_rate,
                testing_set):
    #################### FOLDER SETUP ####################
    MODELS_PATH = "Models"
    RESULT_PATH = "Result"
    REPORT_FOLDER_NAME = "Report"

    """os.makdirs(EXPERIMENT_PATH)
    #os.makedirs(TENSOR_DATA_PATH, exist_ok=True)
    os.mkdir(EXPERIMENT_PATH + "/" + RESULT_PATH)
    os.mkdir(EXPERIMENT_PATH + "/" + MODELS_PATH)"""

    cryptos = os.listdir(DATA_PATH)
    for crypto in cryptos:
        crypto_name = crypto.replace(".csv", "")
        # for each crypto
        # create a folder for data
        os.makedirs(TENSOR_DATA_PATH + "/" + crypto_name, exist_ok=True)
        # create a folder for results
        os.makedirs(EXPERIMENT_PATH + "/" + MODELS_PATH + "/" + crypto_name)
        os.makedirs(EXPERIMENT_PATH + "/" + RESULT_PATH + "/" + crypto_name)

        #QUI PREPARA INPUT PER IL FORECASTING
        data_compliant, features, features_without_date, scaler = \
            prepare_input_forecasting(DATA_PATH + "/" + crypto)

        for temporal, neurons in product(temporal_sequence, number_neurons):
            print(crypto, "\t", temporal, "\t", neurons)
            dataset_tensor = fromtemporal_totensor(np.array(data_compliant), temporal,
                                                               TENSOR_DATA_PATH + "/" + crypto_name + "/",
                                                               crypto_name)


            # DIZIONARIO PER STATISTICHE
            predictions_file = {'symbol': [], 'date': [], 'observed_norm': [], 'predicted_norm': [],
                                'observed_denorm': [],
                                'predicted_denorm': []}
            errors_file = {'symbol': [], 'rmse_norm': [], 'rmse_denorm': []}
            # define a name for this configuration (following folder)
            configuration_name = "LSTM_" + str(neurons) + "_neurons_" + str(temporal) + "_days"
            # Create a folder to save
            # - best model checkpoint
            best_model = "model"
            # - statistics (results)
            statistics = "stats"
            os.mkdir(EXPERIMENT + "/" + MODELS_PATH + "/" + stock_name + "/" + configuration_name)
            os.mkdir(EXPERIMENT + "/" + MODELS_PATH + "/" + stock_name + "/" + configuration_name + "/" + best_model)
            os.mkdir(EXPERIMENT + "/" + RESULT_PATH + "/" + stock_name + "/" + configuration_name)
            os.mkdir(EXPERIMENT + "/" + RESULT_PATH + "/" + stock_name + "/" + configuration_name + "/" + statistics)
            for data_tester in testing_set:
                print("Addestro fino a: ", pd.to_datetime(data_tester))
                train, test = experiments.train_test_split_w_date(features, dataset_tensor, data_tester)
                train = train[:, :, 1:]
                test = test[:, :, 1:]

                x_train, y_train = train[:, :-1, :], train[:, -1, features_without_date.index('Close')]
                x_test, y_test = test[:, :-1, :], test[:, -1, features_without_date.index('Close')]

                # Fare il training
                if data_tester == testing_set[0]:
                    model, history = experiments.train_model(x_train, y_train, x_test, y_test, lstm_neurons=neurons,
                                                             learning_rate=learning_rate,
                                                             dropout=0.2,
                                                             epochs=100,
                                                             batch_size=256,
                                                             dimension_last_layer=1,
                                                             model_path=EXPERIMENT + "/" + MODELS_PATH + "/" + stock_name + "/" + configuration_name + "/" + best_model + "/")
                else:
                    model, history = experiments.train_model(x_train, y_train, x_test, y_test, lstm_neurons=neurons,
                                                             learning_rate=learning_rate,
                                                             dropout=0.2,
                                                             epochs=100,
                                                             batch_size=256, dimension_last_layer=1, model=model,
                                                             model_path=EXPERIMENT + "/" + MODELS_PATH + "/" + stock_name + "/" + configuration_name + "/" + best_model + "/")
                # Tiriamo fuori la predizione per ogni esempio di unused
                test_prediction = model.predict(x_test)
                """ print("Predico per: ", pd.to_datetime(data_tester))
                print("Ho predetto: ", test_prediction)
                print("Valore Reale: ", y_test)
                print("\n")"""
                y_test_denorm = scaler.inverse_transform(y_test.reshape(-1, 1))
                test_prediction_denorm = scaler.inverse_transform(test_prediction)

                y_test = float(y_test)
                test_prediction = float(test_prediction)
                y_test_denorm = float(y_test_denorm)
                test_prediction_denorm = float(test_prediction_denorm)

                # Salvo i risultati nei dizionari
                predictions_file['symbol'].append(stock_name)
                predictions_file['date'].append(data_tester)
                predictions_file['observed_norm'].append(y_test)
                predictions_file['predicted_norm'].append(test_prediction)
                predictions_file['observed_denorm'].append(y_test_denorm)
                predictions_file['predicted_denorm'].append(test_prediction_denorm)


            errors_file['symbol'].append(stock_name)
            rmse = experiments.get_RMSE(predictions_file['observed_norm'], predictions_file['predicted_norm'])
            rmse_denorm = experiments.get_RMSE(predictions_file['observed_denorm'], predictions_file['predicted_denorm'])
            errors_file['rmse_norm'].append(rmse)
            errors_file['rmse_denorm'].append(rmse_denorm)

            pd.DataFrame(data=predictions_file).to_csv(
                EXPERIMENT + "/" + RESULT_PATH + "/" + stock_name + "/" + configuration_name + "/" + statistics + "/" + 'predictions.csv')
            pd.DataFrame(data=errors_file).to_csv(
                EXPERIMENT + "/" + RESULT_PATH + "/" + stock_name + "/" + configuration_name + "/" + statistics + "/" + 'errors.csv')


    #REPORT
    #to_TEST
    """ report_configurations(temporal_sequence_used=temporal_sequence, neurons_used=number_neurons,
                               name_folder_experiment=EXPERIMENT, name_folder_result_experiment=RESULT_PATH,
                               name_folder_report=REPORT_FOLDER_NAME, name_output_files="overall_report")

    report_stockseries(name_folder_experiment=EXPERIMENT, name_folder_result_experiment=RESULT_PATH,
                            name_folder_report=REPORT_FOLDER_NAME,
                            name_files_output="report")"""

    return
