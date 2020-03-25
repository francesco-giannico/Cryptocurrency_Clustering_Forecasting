import os
from decimal import Decimal
from itertools import product
import numpy as np
import pandas as pd
from pandas import DataFrame
from tensorflow.keras.utils import plot_model
from modelling.techniques.forecasting.evaluation.error_measures import get_rmse, get_r_square
from modelling.techniques.forecasting.training.training import prepare_input_forecasting, fromtemporal_totensor, \
    get_training_testing_set, train_model
from utility.computations import get_factors
from utility.folder_creator import folder_creator
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from visualization.line_chart import plot_train_and_validation_loss
import tensorflow_core as tf_core

# np.random.seed(1)
# stable results
tf_core.random.set_seed(2)

PREPROCESSED_PATH = "../preparation/preprocessed_dataset/cleaned/final/"
def single_target(EXPERIMENT_PATH, DATA_PATH, TENSOR_DATA_PATH, window, num_neurons, learning_rate,
                  testing_set, features_to_use, DROPOUT, EPOCHS, PATIENCE,crypto_name):
    #################### FOLDER SETUP ####################
    MODELS_PATH = "models"
    RESULT_PATH = "result"

    try:
        os.remove(EXPERIMENT_PATH + RESULT_PATH + "/merged_predictions.csv")
    except:
        pass

    #crypto_name = crypto.replace(".csv", "")

    # create a folder for data in tensor format
    folder_creator(TENSOR_DATA_PATH + "/" + crypto_name, 0)
    # create a folder for results
    folder_creator(EXPERIMENT_PATH + "/" + MODELS_PATH + "/" + crypto_name, 1)
    folder_creator(EXPERIMENT_PATH + "/" + RESULT_PATH + "/" + crypto_name, 1)


    dataset, features, features_without_date = \
        prepare_input_forecasting(PREPROCESSED_PATH, DATA_PATH, crypto_name, None, features_to_use)


    # print(np.array(dataset)[0]), takes the first row of the dataset (2018-01 2020...etc.)
    dataset_tensor_format = fromtemporal_totensor(np.array(dataset), window,
                                                  TENSOR_DATA_PATH + "/" + crypto_name + "/",
                                                  crypto_name)

    # DICTIONARY FOR STATISTICS
    """predictions_file = {'symbol': [], 'date': [], 'observed_norm': [], 'predicted_norm': [],
                        'observed_denorm': [], 'predicted_denorm': []}
    errors_file = {'symbol': [], 'rmse_norm': [], 'rmse_denorm': []}"""
    predictions_file = {'symbol': [], 'date': [], 'observed_norm': [], 'predicted_norm': []}
    errors_file = {'symbol': [], 'rmse_norm': []}
    """predictions_file = {'symbol': [], 'date': [], 'observed_norm': [], 'predicted_norm': [],
                        'observed_detrans': [], 'predicted_detrans': []}
    errors_file = {'symbol': [], 'rmse_norm': [], 'rmse_detrans': []}"""

    # New folders for this configuration
    configuration_name = "LSTM_" + str(num_neurons) + "_neurons_" + str(window) + "_days"
    # Create a folder to save
    # - best model checkpoint
    # - statistics (results)
    statistics = "stats"
    model_path = EXPERIMENT_PATH + "/" + MODELS_PATH + "/" + crypto_name + "/" + configuration_name + "/"
    results_path = EXPERIMENT_PATH + "/" + RESULT_PATH + "/" + crypto_name + "/" + configuration_name + "/" + statistics + "/"
    folder_creator(model_path, 1)
    folder_creator(results_path, 1)

    train_plot = DataFrame()
    val_plot = DataFrame()
    i = 0
    # starting from the testing set
    for date_to_predict in testing_set:
        # 2 days before the date to predict
        d = pd.to_datetime(date_to_predict) - timedelta(days=1)
        """print('Date to predict: ', date_to_predict)
        print("Training until: ", d)"""
        """the format of train and test is the following one:
         [
            [[Row1],[Row2]],
            [[Row1],[Row2]],
            ....
            [[Row1],[Row2]],
         ]
        thus for element accessing there are the following three indexes:
          1)e.g [[items],[items]]
          2)e.g [items],[items]
          3)e.g items
        """
        train, test = get_training_testing_set(dataset_tensor_format, date_to_predict)

        # ['2018-01-01' other numbers separated by comma],it removes the date.
        train = train[:, :, 1:]
        test = test[:, :, 1:]

        index_of_target_feature = features_without_date.index('Close')
        # print(index_of_target_feature)
        # remove the last day before the day to predict:
        # e.g date to predict 2019-01-07 thus the data about 2019-01-06 will be discarded.
        # e.g [[items],[items2],[items3]] becames [[items1],[items2]]
        # also, i will remove the "Close" feature, thanks to the third index (1)
        # x_train= train[:, :-1, index_of_target_feature:]
        x_train = train[:, :-1, :]
        """print(x_train.shape)
        print(x_train[0])"""
        # remove the last day before the day to predict, by doing -1
        # returns an array with all the values of the feature close
        # this contains values about the target feature!
        y_train = train[:, -1, index_of_target_feature]
        # print(y_train)

        # NOTE: in the testing set we must have the dates to evaluate the experiment without the date to forecast!!!
        # remove the day to predict
        # e.g date to predict 2019-01-07 thus the data about 2019-01-07 will be discarded.
        # e.g [[items],[items2],[items3]] becames [[items1],[items2]]
        # x_test = test[:, :-1, index_of_target_feature:]
        x_test = test[:, :-1, :]
        # remove the last day before the day to predict, by doing -1
        # returns an array with all the values of the feature close to predict!
        y_test = test[:, -1, index_of_target_feature]

        # change the data type, from object to float
        # print(x_train[0][0])
        x_train = x_train.astype('float')
        # print(x_train[0][0])
        y_train = y_train.astype('float')
        x_test = x_test.astype('float')
        y_test = y_test.astype('float')

        """print(x_train.shape[0])
        print(get_factors(x_train.shape[0]))"""
        #batch size must be a factor of the number of training elements
        factors = get_factors(x_train.shape[0])
        if len(factors) == 2:
            BATCH_SIZE = factors[1]
            #print("only 2 elements " + str(BATCH_SIZE))
        elif len(factors) % 2 != 0:  # odd
            BATCH_SIZE = int(np.median(factors))
            #print("is odd: " + str(BATCH_SIZE))
        else:  # even
            # vectors=np.split(np.asarray(factors),2)
            # print(vectors)
            # BATCH_SIZE=vectors[1][0]
            BATCH_SIZE = factors[(len(factors) - 2)]
            # BATCH_SIZE=int(np.median(factors))

            # BATCH_SIZE=np.min(factors[2:(len(factors)-2)])
            #print("Batch used " + str(BATCH_SIZE))

        # if the date to predict is the first date in the testing_set
        if date_to_predict == testing_set[0]:
            model, history = train_model(x_train, y_train, x_test, y_test,
                                         num_neurons=num_neurons,
                                         learning_rate=learning_rate,
                                         dropout=DROPOUT,
                                         epochs=EPOCHS,
                                         batch_size=BATCH_SIZE,
                                         patience=PATIENCE,
                                         dimension_last_layer=1,
                                         model_path=model_path)
            # information about neural network created
            plot_model(model, to_file=model_path + "neural_network.png", show_shapes=True,
                       show_layer_names=True, expand_nested=True, dpi=150)

        else:
            model, history = train_model(x_train, y_train, x_test, y_test,
                                         num_neurons=num_neurons,
                                         learning_rate=learning_rate,
                                         dropout=DROPOUT,
                                         epochs=EPOCHS,
                                         batch_size=BATCH_SIZE,
                                         patience=PATIENCE,
                                         dimension_last_layer=1,
                                         model=model,
                                         model_path=model_path)

        train_plot[str(i)] = pd.Series(history.history['loss'])
        val_plot[str(i)] = pd.Series(history.history['val_loss'])
        i += 1

        # Predict for each date in the validation set
        test_prediction = model.predict(x_test, use_multiprocessing=True)

        # denormalization
        # reshape(-1,1) means that you are not specifing only the column dimension, whist the row dimension is unknown
        """y_test_denorm = scaler_target_feature.inverse_transform(y_test.reshape(-1, 1))
        test_prediction_denorm = scaler_target_feature.inverse_transform(test_prediction)"""

        # detransformation
        """" y_test_detransformed = qt.inverse_transform(y_test_denorm.reshape(-1, 1))
        test_prediction_detransformed = qt.inverse_transform(test_prediction_denorm)"""

        # changing data types
        # normalized
        y_test = float(y_test)
        test_prediction = float(test_prediction)
        # denormalized
        """y_test_denorm = float(y_test_denorm)
        test_prediction_denorm = float(test_prediction_denorm)"""

        # detransformed
        """y_test_detr = float(y_test_detransformed)
        test_prediction_detr = float(test_prediction_detransformed)"""
        print("Num of entries for training: ", x_train.shape[0])
        # print("Num of element for validation: ", x_test.shape[0])
        print("Training until: ", d)
        print("Predicting for: ", date_to_predict)
        print("Predicted: ", test_prediction)
        print("Actual: ", y_test)
        print("\n")
        """
        print("Predicting for: ", date_to_predict)
        print("Predicted: ", test_prediction_detr)
        print("Actual: ", y_test_detr)
        print("\n")
        """

        # Saving the predictions on the dictionaries
        predictions_file['symbol'].append(crypto_name)
        predictions_file['date'].append(date_to_predict)
        predictions_file['observed_norm'].append(y_test)
        predictions_file['predicted_norm'].append(test_prediction)
        """predictions_file['observed_denorm'].append(y_test_denorm)
        predictions_file['predicted_denorm'].append(test_prediction_denorm)"""
        """predictions_file['observed_detrans'].append(y_test_detr)
        predictions_file['predicted_detrans'].append(test_prediction_detr)"""

    # Plot training & validation loss values
    plot_train_and_validation_loss(train_plot, val_plot, model_path)

    # Saving the RMSE into the dictionaries
    errors_file['symbol'].append(crypto_name)

    # normalized RMSE
    rmse = get_rmse(predictions_file['observed_norm'], predictions_file['predicted_norm'])
    errors_file['rmse_norm'].append(rmse)

    # denormalized RMSE
    # rmse_denorm = get_rmse(predictions_file['observed_denorm'], predictions_file['predicted_denorm'])
    # errors_file['rmse_denorm'].append(rmse_denorm)

    # detransf RMSE
    """rmse_detrans= get_rmse(predictions_file['observed_detrans'], predictions_file['predicted_detrans'])
    errors_file['rmse_detrans'].append(rmse_detrans)"""
    # serialization
    pd.DataFrame(data=predictions_file).to_csv(results_path + 'predictions.csv', index=False)
    pd.DataFrame(data=errors_file).to_csv(results_path + 'errors.csv', index=False)

    return









