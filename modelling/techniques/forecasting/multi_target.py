
from datetime import timedelta
import numpy as np
import pandas as pd
from itertools import product
from pandas import DataFrame
from tensorflow_core.python.keras.utils.vis_utils import plot_model

from modelling.techniques.forecasting.evaluation.error_measures import get_rmse
from modelling.techniques.forecasting.training.training import prepare_input_forecasting, fromtemporal_totensor, \
    get_training_validation_testing_set, train_model
from utility.computations import get_factors
from utility.folder_creator import folder_creator
from visualization.line_chart import plot_train_and_validation_loss
import tensorflow_core as tf_core

from tensorflow.keras import backend as K
#np.random.seed(0)
tf_core.random.set_seed(2)

# features_to_exclude_from_scaling = ['Symbol_1','Symbol_2','Symbol_3','Symbol_4','Symbol_5','Symbol_6','Symbol_7','Symbol_8']

PREPROCESSED_PATH="../preparation/preprocessed_dataset/cleaned/final/"
def multi_target(EXPERIMENT_PATH, DATA_PATH, TENSOR_DATA_PATH,
                        window_sequences, list_num_neurons,
                        learning_rate,
                        dimension_last_layer,
                        testing_set,
                        cryptos,
                        features_to_use, DROPOUT, EPOCHS, PATIENCE,
                        number_of_days_to_predict,start_date,end_date):

    #################### FOLDER SETUP ####################
    MODELS_PATH = "models"
    RESULT_PATH = "result"

    horizontal_file="horizontal"
    horizontal_name=horizontal_file.replace(".csv","")

    # create a folder for data in tensor format
    folder_creator(TENSOR_DATA_PATH + "/" + horizontal_name, 0)
    # create a folder for models
    folder_creator(EXPERIMENT_PATH  + MODELS_PATH + "/", 0)
    folder_creator(EXPERIMENT_PATH + "/" + RESULT_PATH + "/", 0)

    dataset, features, features_without_date = \
        prepare_input_forecasting(PREPROCESSED_PATH, DATA_PATH, horizontal_file,start_date,end_date,cryptos,features_to_use)

    #takes all the target
    indexes_of_target_features = [features_without_date.index(f) for f in features_without_date if
                                   f.startswith('Close')]

    # [(30, 128), (30, 256), (100, 128), (100, 256), (200, 128), (200, 256)]
    # print(np.array(dataset)[0]), takes the first row of the dataset (2018-01 2020...etc.)
    for window, num_neurons in product(window_sequences, list_num_neurons):
        print('Current configuration: ')
        print("Crypto_symbol: ", horizontal_file, "\t", "Window_sequence: ", window, "\t", "Neurons: ", num_neurons)

        # non prende in input i neuroni questo.
        dataset_tensor_format = fromtemporal_totensor(np.array(dataset), window,
                                                      TENSOR_DATA_PATH + "/" + horizontal_name + "/",
                                                      horizontal_name)


        #DICTIONARY FOR STATISTICS
        predictions_file = {'date': []}
        for crypto in cryptos:
            crypto= str(crypto)
            predictions_file[crypto+ "_observed_norm"] = []
            predictions_file[crypto + "_predicted_norm"] = []
            #predictions_file[crypto + "_observed_denorm"] = []
            #predictions_file[crypto + "_predicted_denorm"] = []

        #New folders for this configuration
        configuration_name = "LSTM_" + str(num_neurons) + "_neurons_" + str(window) + "_days"
        # Create a folder to save
        # - best model checkpoint
        # - statistics (results)
        statistics = "stats"
        model_path = EXPERIMENT_PATH + MODELS_PATH + "/" + configuration_name + "/"
        #todo
        #results_path = EXPERIMENT_PATH + "/" + RESULT_PATH + "/" + horizontal_name + "/" + configuration_name + "/" + statistics +"/"
        folder_creator(model_path,0)
        #todo
        #folder_creator(results_path,1)

        """train_plot = DataFrame()
        val_plot = DataFrame()"""
        i = 0
        # starting from the testing set
        for date_to_predict in testing_set:
            # 2 days before the date to predict
            """d = pd.to_datetime(date_to_predict) - timedelta(days=2)
            print('Date to predict: ', date_to_predict)
            print("Training until: ", d)"""
        
            train, test = get_training_validation_testing_set(dataset_tensor_format, date_to_predict,number_of_days_to_predict)

            # ['2018-01-01' other numbers separated by comma],it removes the date.
            train = train[:, :, 1:]
            test = test[:, :, 1:]
            # remove the last day before the day to predict:
            # e.g date to predict 2019-01-07 thus the data about 2019-01-06 will be discarded.
            # e.g [[items],[items2],[items3]] becames [[items1],[items2]]
            x_train = train[:, :-1, :]

            # remove the last day before the day to predict, by doing -1
            # returns an array with all the values of the feature close
            y_train = train[:, -1, indexes_of_target_features]

            # NOTE: in the testing set we must have the dates to evaluate the experiment without the date to forecast!!!
            # remove the day to predict
            # e.g date to predict 2019-01-07 thus the data about 2019-01-07 will be discarded.
            # e.g [[items],[items2],[items3]] becames [[items1],[items2]]
            x_test = test[:, :-1, :]
            # remove the last day before the day to predict, by doing -1
            # returns an array with all the values of the feature close to predict!
            y_test = test[:, -1, indexes_of_target_features]

            # change the data type, from object to float
            x_train = x_train.astype('float')
            y_train = y_train.astype('float')
            x_test = x_test.astype('float')
            y_test = y_test.astype('float')

            BATCH_SIZE=x_train.shape[0]
            # if the date to predict is the first date in the testing_set
            #if date_to_predict == testing_set[0]:
            model, history = train_model(x_train, y_train,
                                         num_neurons=num_neurons,
                                         learning_rate=learning_rate,
                                         dropout=DROPOUT,
                                         epochs=EPOCHS,
                                         batch_size=BATCH_SIZE,
                                         dimension_last_layer=dimension_last_layer,
                                         date_to_predict=date_to_predict,
                                         model_path=model_path,patience=PATIENCE)
            # information about neural network created
            plot_model(model, to_file=model_path + "neural_network.png", show_shapes=True,
                       show_layer_names=True, expand_nested=True, dpi=150)
            # this is important!!
            K.clear_session()
            filename = "model_train_val_loss_bs_" + str(BATCH_SIZE) + "_target_" + str(date_to_predict)
            plot_train_and_validation_loss(pd.Series(history.history['loss']), pd.Series(history.history['val_loss']),
                                           model_path, filename)

            # Predict for each date in the validation set
            test_prediction = model.predict(x_test)

            print("Num of entries for training: ", x_train.shape[0])
            # print("Num of element for validation: ", x_test.shape[0])
            # print("Training until: ", pd.to_datetime(date_to_predict) - timedelta(days=3))
            days = []
            i = number_of_days_to_predict-1
            while i > 0:
                d = pd.to_datetime(date_to_predict) - timedelta(days=i)
                days.append(d)
                i -= 1
            days.append(pd.to_datetime(date_to_predict))


            i = 0
            for d in days:
                print("Predicting for: ", d)
                print("Predicted: ", test_prediction[i])
                print("Actual: ", y_test[i])
                i+=1
            print("\n")

            # Saving the predictions on the dictionarie
            i = 0
            for d in days:
                predictions_file['date'].append(d)
                for crypto, observed in zip(cryptos, y_test[i]):
                    predictions_file[crypto + "_observed_norm"].append(float(observed))
                for crypto,predicted in zip(cryptos, test_prediction[i]):
                    predictions_file[crypto + "_predicted_norm"].append(float(predicted))
                i+=1
            #break

        #divides results by crypto.
        for crypto in cryptos:
            PATH_CRYPTO=EXPERIMENT_PATH + "/" + RESULT_PATH + "/" + crypto+"/"+configuration_name+"/"+statistics+"/"
            folder_creator(PATH_CRYPTO, 0)

            crypto_prediction_file = {}
            crypto_prediction_file['date'] = predictions_file['date']
            crypto_prediction_file['observed_norm'] = predictions_file[crypto + '_observed_norm']
            crypto_prediction_file['predicted_norm'] = predictions_file[crypto + '_predicted_norm']

            crypto_errors_file = {'symbol': [], 'rmse_norm': []}
            crypto_errors_file['symbol'].append(crypto)
            rmse = get_rmse(crypto_prediction_file['observed_norm'], crypto_prediction_file['predicted_norm'])
            #rmse_denorm = get_rmse(crypto_prediction_file['observed_denorm'], crypto_prediction_file['predicted_denorm'])
            crypto_errors_file['rmse_norm'].append(rmse)
            #crypto_errors_file['rmse_denorm'].append(rmse_denorm)

            #i'm just taking 3 dates at time and computing the RMSE over these three dates
            #eventually, i'll do the average of these rmses
            rmses=[]
            crypto_errors_file_1 = {'symbol': [], 'rmse_norm': []}
            crypto_errors_file_1['symbol'].append(crypto)

            """try:
                min = 0
                max = number_of_days_to_predict
                print(crypto_prediction_file['observed_norm'])
                print(crypto_prediction_file['observed_norm'][0:2])
                while max <= len(crypto_prediction_file['observed_norm']):
                    rmses.append(get_rmse(crypto_prediction_file['observed_norm'][min:max],
                                          crypto_prediction_file['predicted_norm'][min:max]))
                    min = max
                    max += number_of_days_to_predict
                avg_rmse = (1 / len(rmses)) * np.sum(rmses)
                crypto_errors_file_1['rmse_norm'].append(avg_rmse)
            except:
                rmse=get_rmse(crypto_prediction_file['observed_norm'],
                                          crypto_prediction_file['predicted_norm'])
                crypto_errors_file_1['rmse_norm'].append(rmse)"""

            #serialization
            pd.DataFrame(data=crypto_prediction_file).to_csv(PATH_CRYPTO + 'predictions.csv',index=False)
            pd.DataFrame(data=crypto_errors_file).to_csv(PATH_CRYPTO + 'errors.csv',index=False)
            #pd.DataFrame(data=crypto_errors_file_1).to_csv(PATH_CRYPTO + 'errors_1.csv', index=False)

    return
