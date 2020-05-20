import os
from decimal import Decimal
from itertools import product
import numpy as np
import pandas as pd
from pandas import DataFrame
from tensorflow.keras.utils import plot_model
from tensorflow_core.python.keras.utils.np_utils import to_categorical

from modelling.techniques.forecasting.evaluation.error_measures import get_rmse,  \
    get_classification_stats
from modelling.techniques.forecasting.training.training import prepare_input_forecasting, fromtemporal_totensor, \
    get_training_validation_testing_set, train_single_target_model
from utility.computations import get_factors
from utility.folder_creator import folder_creator
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from visualization.line_chart import plot_train_and_validation_loss, plot_train_and_validation_accuracy
import tensorflow_core as tf_core
import time
import random as rn
from tensorflow.keras import backend as K
np.random.seed(42)
rn.seed(42)
# stable results
tf_core.random.set_seed(42)


PREPROCESSED_PATH = "../preparation/preprocessed_dataset/cleaned/final/"
def single_target(EXPERIMENT_PATH, DATA_PATH, TENSOR_DATA_PATH, window_sequences, list_num_neurons, learning_rate,
                  testing_set, features_to_use, DROPOUT, EPOCHS, PATIENCE,start_date,end_date):


    #################### FOLDER SETUP ####################
    MODELS_PATH = "models"
    RESULT_PATH = "result"
    TIME_PATH="time"
    # starting from the testing set
    for window, num_neurons in product(window_sequences, list_num_neurons):
        print('Current configuration: ')
        print("Window_sequence: ", window, "\t", "Neurons: ", num_neurons)
        crypto_name=""
        predictions_file = {'symbol': [], 'date': [], 'observed_class': [], 'predicted_class': []}
        macro_avg_recall_file = {'symbol': [], 'macro_avg_recall': []}
        results_path=""
        for dataset_name in os.listdir(DATA_PATH):
            #format of dataset name: Crypto_DATE_TO_PREDICT.csv
            splitted=dataset_name.split("_")
            if(crypto_name!=splitted[0]):#new crypto

                # DICTIONARY FOR STATISTICS
                # Saving the accuracy into the dictionaries
                try:
                    macro_avg_recall_file['symbol'].append(crypto_name)
                    # accuracy
                    performances = get_classification_stats(predictions_file['observed_class'],
                                                            predictions_file['predicted_class'])
                    macro_avg_recall_file['macro_avg_recall'].append(performances.get('macro avg').get('recall'))

                    # serialization
                    pd.DataFrame(data=predictions_file).to_csv(results_path + 'predictions.csv', index=False)
                    pd.DataFrame(data=macro_avg_recall_file).to_csv(results_path + 'macro_avg_recall.csv', index=False)
                except:
                    pass
                #update the name
                crypto_name = splitted[0]
                # clean the dictionary
                predictions_file = {'symbol': [], 'date': [], 'observed_class': [], 'predicted_class': []}
                macro_avg_recall_file = {'symbol': [], 'macro_avg_recall': []}

            print("Current crypto: ", crypto_name, "\t")
            date_to_predict=str(splitted[1]).replace(".csv","")

            # create a folder for data in tensor format
            folder_creator(TENSOR_DATA_PATH + "/" + crypto_name, 0)
            # create a folder for results
            folder_creator(EXPERIMENT_PATH + "/" + MODELS_PATH + "/" + crypto_name, 0)
            folder_creator(EXPERIMENT_PATH + "/" + RESULT_PATH + "/" + crypto_name, 0)
            # create folder for time spent
            folder_creator(EXPERIMENT_PATH + "/" + TIME_PATH + "/" + crypto_name, 0)

            dataset, features, features_without_date = \
                prepare_input_forecasting(PREPROCESSED_PATH, DATA_PATH, dataset_name,start_date,end_date,None, features_to_use)

            # print(np.array(dataset)[0]), takes the first row of the dataset (2018-01 2020...etc.)
            dataset_tensor_format = fromtemporal_totensor(np.array(dataset), window,
                                                          TENSOR_DATA_PATH + "/" + crypto_name + "/",
                                                          crypto_name)
            # New folders for this configuration
            configuration_name = "LSTM_" + str(num_neurons) + "_neurons_" + str(window) + "_days"
            # Create a folder to save
            # - best model checkpoint
            # - statistics (results)
            statistics = "stats"
            model_path = EXPERIMENT_PATH + "/" + MODELS_PATH + "/" + crypto_name + "/" + configuration_name + "/"
            results_path = EXPERIMENT_PATH + "/" + RESULT_PATH + "/" + crypto_name + "/" + configuration_name + "/" + statistics + "/"
            folder_creator(model_path, 0)
            folder_creator(results_path,0)

            #train, validation,test = get_training_validation_testing_set(dataset_tensor_format, date_to_predict)
            train, test = get_training_validation_testing_set(dataset_tensor_format, date_to_predict)
            # ['2018-01-01' other numbers separated by comma],it removes the date.

            train = train[:, :, 1:]
            test = test[:, :, 1:]

            index_of_target_feature = features_without_date.index('trend')

            x_train = train[:, :-1, :index_of_target_feature]
            """print("X_TRAIN")
            print(x_train)
            print(x_train.shape)"""

            y_train = train[:, -1, index_of_target_feature]
            """print("Y_TRAIN")
            print(y_train)
            print(y_train.shape)"""

            x_test = test[:, :-1, :index_of_target_feature]
            """print("X_TEST")
            print(x_test)
            print(x_test.shape)"""

            y_test = test[:, -1, index_of_target_feature]
            """print("Y_TEST")
            print(y_test)
            print(y_test.shape)"""

            # change the data type, from object to float
            x_train = x_train.astype('float')
            # print(x_train[0][0])
            #y_train = y_train.astype('float')
            x_test = x_test.astype('float')
            #y_test = y_test.astype('float')
            #print(y_test)
            # one hot encode y
            y_train  = to_categorical(y_train)
            y_test = to_categorical(y_test)
            """print(y_train)
            print(y_test)"""
            #batch size must be a factor of the number of training elements
            BATCH_SIZE=x_train.shape[0]
            # if the date to predict is the first date in the testing_set
            #if date_to_predict == testing_set[0]:
            model, history = train_single_target_model(x_train, y_train,
                                         num_neurons=num_neurons,
                                         learning_rate=learning_rate,
                                         dropout=DROPOUT,
                                         epochs=EPOCHS,
                                         batch_size=BATCH_SIZE,
                                         patience=PATIENCE,
                                         num_categories=len(y_train[0]),
                                         date_to_predict=date_to_predict,
                                         model_path=model_path)
            # plot neural network's architecture
            plot_model(model, to_file=model_path + "neural_network.png", show_shapes=True,
                       show_layer_names=True, expand_nested=True, dpi=150)

            #plot loss
            filename="model_train_val_loss_bs_"+str(BATCH_SIZE)+"_target_"+str(date_to_predict)
            plot_train_and_validation_loss(pd.Series(history.history['loss']),pd.Series(history.history['val_loss']),model_path,filename)

            #plot accuracy
            filename = "model_train_val_accuracy_bs_" + str(BATCH_SIZE) + "_target_" + str(date_to_predict)
            plot_train_and_validation_accuracy(pd.Series(history.history['accuracy']),
                                           pd.Series(history.history['val_accuracy']), model_path, filename)

            # Predict for each date in the validation set
            test_prediction = model.predict(x_test)
            # this is important!!
            K.clear_session()
            tf_core.random.set_seed(42)

            # changing data types
            #test_prediction = float(test_prediction)
            #test_prediction=test_prediction.astype("float")

            print("Num of entries for training: ", x_train.shape[0])
            # print("Num of element for validation: ", x_test.shape[0])
            #print("Training until: ", pd.to_datetime(date_to_predict) - timedelta(days=3))

            # invert encoding: argmax of numpy takes the higher value in the array
            print("Predicting for: ", date_to_predict)
            print("Predicted: ", np.argmax(test_prediction[0]))
            print("Actual: ", np.argmax(y_test[0]))
            print("\n")

            # Saving the predictions on the dictionarie
            predictions_file['symbol'].append(crypto_name)
            predictions_file['date'].append(date_to_predict)
            predictions_file['observed_class'].append(np.argmax(y_test[0]))
            predictions_file['predicted_class'].append(np.argmax(test_prediction[0]))

    return
