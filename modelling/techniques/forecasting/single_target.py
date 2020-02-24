import os
from itertools import product
import numpy as np
import pandas as pd

from modelling.techniques.forecasting.evaluation.rmse import get_RMSE
from modelling.techniques.forecasting.training.training import prepare_input_forecasting, fromtemporal_totensor, \
    get_training_testing_set, train_model
from utility.folder_creator import folder_creator
from datetime import datetime, timedelta
np.random.seed(0)

# TENSOR_PATH = "../crypto_TensorData"
# Parameters of experiments
# temporal_sequence_considered = [30, 100, 200]
# number_neurons_LSTM = [128, 256]
# features_to_exclude_from_scaling = ['Symbol']

PREPROCESSED_PATH="../preparation/preprocessed_dataset/cleaned/final/"
def single_target(EXPERIMENT_PATH, DATA_PATH, TENSOR_DATA_PATH, window_sequence, num_neurons, learning_rate,
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

        dataset, features, features_without_date, scaler_target_feature=  \
            prepare_input_forecasting(PREPROCESSED_PATH,DATA_PATH,crypto)

        # DICTIONARY FOR STATISTICS
        predictions_file = {'symbol': [], 'date': [], 'observed_norm': [], 'predicted_norm': [],
                            'observed_denorm': [], 'predicted_denorm': []}
        errors_file = {'symbol': [], 'rmse_norm': [], 'rmse_denorm': []}

        #print(list(product(temporal_sequence, number_neurons)))
        #[(30, 128), (30, 256), (100, 128), (100, 256), (200, 128), (200, 256)]
        #print(np.array(dataset)[0]), takes the first row of the dataset (2018-01 2020...etc.)
        for window, num_neurons in product(window_sequence, num_neurons):
            print('Current configuration: ')
            print("Crypto_symbol: ",crypto,"\t", "Window_sequence: ",window,"\t", "Neurons: ",num_neurons)

            #non prende in input i neuroni questo.
            dataset_tensor_format = fromtemporal_totensor(np.array(dataset), window,
                                                   TENSOR_DATA_PATH + "/" + crypto_name + "/",
                                                   crypto_name)



            #New folders for this configuration
            configuration_name = "LSTM_" + str(num_neurons) + "_neurons_" + str(window) + "_days"
            # Create a folder to save
            # - best model checkpoint
            # - statistics (results)
            best_model = "model"
            statistics = "stats"
            model_path = EXPERIMENT_PATH + "/" + MODELS_PATH + "/" + crypto_name + "/" + configuration_name + "/" + best_model+"/"
            results_path = EXPERIMENT_PATH + "/" + RESULT_PATH + "/" + crypto_name + "/" + configuration_name + "/" + statistics +"/"
            folder_creator(model_path,1)
            folder_creator(results_path,1)


            #starting from the testing set
            for date_to_predict in testing_set:
                #todo-balla, in realtà addestro fino a 2 giorni prima della data da predire.

                d = pd.to_datetime(date_to_predict) - timedelta(days=2)
                print('Date to predict: ', date_to_predict)
                print("Training until: ", d)
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

                # change the data type, from object to float
                x_train = x_train.astype('float')
                y_train = y_train.astype('float')
                x_test = x_test.astype('float')
                y_test = y_test.astype('float')



                #x_train= tf.convert_to_tensor(x_train, np.float32)
                #if the date to predict is the first date in the testing_set
                if date_to_predict == testing_set[0]:
                    model, history = train_model(x_train,y_train,x_test,y_test, num_neurons=num_neurons,
                                                             learning_rate=learning_rate,
                                                             dropout=0.2,
                                                             epochs=100,
                                                             batch_size=250,
                                                             dimension_last_layer=1,
                                                             model_path=model_path)
                else:
                    model, history = train_model(x_train, y_train, x_test, y_test, num_neurons=num_neurons,
                                                             learning_rate=learning_rate,
                                                             dropout=0.2,
                                                             epochs=100,
                                                             batch_size=256, dimension_last_layer=1, model=model,
                                                             model_path=model_path)


                #Predict for each date in the validation set
                test_prediction = model.predict(x_test)

                print("Predicting for: ", date_to_predict)
                print("Predicted: ", test_prediction)
                print("Actual: ", y_test)

                #denormalization
                #reshape(-1,1) means that you are not specifing only the column dimension, whist the row dimension is unknown
                y_test_denorm = scaler_target_feature.inverse_transform(y_test.reshape(-1, 1))
                test_prediction_denorm = scaler_target_feature.inverse_transform(test_prediction)
            
                #changing data types
                #normalized
                y_test = float(y_test)
                test_prediction = float(test_prediction)
                #denormalized
                y_test_denorm = float(y_test_denorm)
                test_prediction_denorm = float(test_prediction_denorm)

                #Saving the predictions on the dictionaries
                predictions_file['symbol'].append(crypto_name)
                predictions_file['date'].append(date_to_predict)
                predictions_file['observed_norm'].append(y_test)
                predictions_file['predicted_norm'].append(test_prediction)
                predictions_file['observed_denorm'].append(y_test_denorm)
                predictions_file['predicted_denorm'].append(test_prediction_denorm)

            #Saving the RMSE into the dictionaries
            errors_file['symbol'].append(crypto_name)

            #normalized RMSE
            rmse = get_RMSE(predictions_file['observed_norm'], predictions_file['predicted_norm'])
            errors_file['rmse_norm'].append(rmse)

            #denormalized RMSE
            rmse_denorm = get_RMSE(predictions_file['observed_denorm'], predictions_file['predicted_denorm'])
            errors_file['rmse_denorm'].append(rmse_denorm)

            #serialization
            pd.DataFrame(data=predictions_file).to_csv(results_path + 'predictions.csv',index=False)
            pd.DataFrame(data=errors_file).to_csv(results_path  + 'errors.csv',index=False)
        break


#REPORT
#to_TEST
""" report_configurations(temporal_sequence_used=temporal_sequence, neurons_used=number_neurons,
                           name_folder_experiment=EXPERIMENT, name_folder_result_experiment=RESULT_PATH,
                           name_folder_report=REPORT_FOLDER_NAME, name_output_files="overall_report")

report_stockseries(name_folder_experiment=EXPERIMENT, name_folder_result_experiment=RESULT_PATH,
                        name_folder_report=REPORT_FOLDER_NAME,
                        name_files_output="report")"""


