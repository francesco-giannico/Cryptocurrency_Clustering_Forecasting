
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
                        features_to_use, DROPOUT, EPOCHS, PATIENCE):

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
        prepare_input_forecasting(PREPROCESSED_PATH, DATA_PATH, horizontal_file,cryptos,features_to_use)

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
        
            train, test = get_training_validation_testing_set(dataset_tensor_format, date_to_predict)

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
            """ factors = get_factors(x_train.shape[0])
            if len(factors) == 2:
                BATCH_SIZE = factors[1]
                # print("only 2 elements " + str(BATCH_SIZE))
            elif len(factors) % 2 != 0:  # odd
                BATCH_SIZE = int(np.median(factors))
                # print("is odd: " + str(BATCH_SIZE))
            else:  # even
                # vectors=np.split(np.asarray(factors),2)
                # print(vectors)
                # BATCH_SIZE=vectors[1][0]
                BATCH_SIZE = factors[(len(factors) - 2)]
                # BATCH_SIZE=int(np.median(factors))

                # BATCH_SIZE=np.min(factors[2:(len(factors)-2)])
                # print("Batch used " + str(BATCH_SIZE))"""
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
            """else:
                model, history = train_model(x_train, y_train,
                                             num_neurons=num_neurons,
                                             learning_rate=learning_rate,
                                             dropout=DROPOUT,
                                             epochs=EPOCHS,
                                             batch_size=BATCH_SIZE,
                                             dimension_last_layer=dimension_last_layer,
                                             date_to_predict=date_to_predict,
                                             model=model,
                                             model_path=model_path,patience=PATIENCE)"""

            filename = "model_train_val_loss_bs_" + str(BATCH_SIZE) + "_target_" + str(date_to_predict)
            plot_train_and_validation_loss(pd.Series(history.history['loss']), pd.Series(history.history['val_loss']),
                                           model_path, filename)
            """train_plot[str(i)] = pd.Series(history.history['loss'])
            val_plot[str(i)] = pd.Series(history.history['val_loss'])"""
            i += 1

            # Predict for each date in the validation set
            test_prediction = model.predict(x_test)
            print("Num of entries for training: ", x_train.shape[0])
            # print("Num of element for validation: ", x_test.shape[0])
            print("Training until: ", pd.to_datetime(date_to_predict) - timedelta(days=3))

            d1 = pd.to_datetime(date_to_predict) - timedelta(days=2)
            d2 = pd.to_datetime(date_to_predict) - timedelta(days=1)

            print("Predicting for: ", d1)
            print("Predicted: ", test_prediction[0])
            print("Actual: ", y_test[0])

            print("Predicting for: ", d2)
            print("Predicted: ", test_prediction[1])
            print("Actual: ", y_test[1])

            print("Predicting for: ", date_to_predict)
            print("Predicted: ", test_prediction[2])
            print("Actual: ", y_test[2])

            print("\n")

            """print("Predicting for: ", date_to_predict)
            print("Predicted: ", test_prediction[0])
            print("Actual: ", y_test)"""

            """y_test_denorm = scaler_target_feature.inverse_transform(y_test.reshape(-1, dimension_last_layer))
            test_prediction_denorm = scaler_target_feature.inverse_transform(test_prediction)
                """
            # Saving the predictions on the dictionaries
            predictions_file['date'].append(d1)
            for crypto, observed in zip(cryptos, y_test[0]):
                predictions_file[crypto + "_observed_norm"].append(float(observed))
            for crypto,predicted in zip(cryptos, test_prediction[0]):
                predictions_file[crypto + "_predicted_norm"].append(float(predicted))

            predictions_file['date'].append(d2)
            for crypto, observed in zip(cryptos, y_test[1]):
                predictions_file[crypto + "_observed_norm"].append(float(observed))
            for crypto, predicted in zip(cryptos, test_prediction[1]):
                predictions_file[crypto + "_predicted_norm"].append(float(predicted))

            predictions_file['date'].append(date_to_predict)
            for crypto, observed in zip(cryptos, y_test[2]):
                predictions_file[crypto + "_observed_norm"].append(float(observed))
            for crypto, predicted in zip(cryptos, test_prediction[2]):
                predictions_file[crypto + "_predicted_norm"].append(float(predicted))

            """for crypto, observed in zip(cryptos, y_test_denorm[0]):
                predictions_file[crypto + "_observed_denorm"].append(float(observed))

            for crypto,predicted in zip(cryptos, test_prediction_denorm[0]):
                predictions_file[crypto + "_predicted_denorm"].append(float(predicted))"""
            #break


        # Plot training & validation loss values
        #plot_train_and_validation_loss(train_plot, val_plot, model_path)


        #divides results by crypto.
        for crypto in cryptos:
            PATH_CRYPTO=EXPERIMENT_PATH + "/" + RESULT_PATH + "/" + crypto+"/"+configuration_name+"/"+statistics+"/"
            folder_creator(PATH_CRYPTO, 0)

            crypto_prediction_file = {}
            crypto_prediction_file['date'] = predictions_file['date']
            crypto_prediction_file['observed_norm'] = predictions_file[crypto + '_observed_norm']
            crypto_prediction_file['predicted_norm'] = predictions_file[crypto + '_predicted_norm']
            """crypto_prediction_file['observed_denorm'] = predictions_file[crypto + '_observed_denorm']
            crypto_prediction_file['predicted_denorm'] = predictions_file[crypto + '_predicted_denorm']
"""
            #crypto_errors_file = {'symbol': [], 'rmse_norm': [], 'rmse_denorm': []}
            crypto_errors_file = {'symbol': [], 'rmse_norm': []}
            crypto_errors_file['symbol'].append(crypto)
            rmse = get_rmse(crypto_prediction_file['observed_norm'], crypto_prediction_file['predicted_norm'])
            #rmse_denorm = get_rmse(crypto_prediction_file['observed_denorm'], crypto_prediction_file['predicted_denorm'])
            crypto_errors_file['rmse_norm'].append(rmse)
            #crypto_errors_file['rmse_denorm'].append(rmse_denorm)

            #serialization
            pd.DataFrame(data=crypto_prediction_file).to_csv(PATH_CRYPTO + 'predictions.csv',index=False)
            pd.DataFrame(data=crypto_errors_file).to_csv(PATH_CRYPTO + 'errors.csv',index=False)

    return
