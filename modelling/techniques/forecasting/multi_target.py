
from datetime import timedelta
import numpy as np
import pandas as pd
from itertools import product
from tensorflow.keras.utils import plot_model
from tensorflow.keras.utils import to_categorical
from modelling.techniques.forecasting.evaluation.error_measures import get_classification_stats
from modelling.techniques.forecasting.training.training import prepare_input_forecasting, fromtemporal_totensor, \
    get_training_validation_testing_set, train_multi_target_model
from utility.folder_creator import folder_creator
from visualization.line_chart import plot_train_and_validation_loss, plot_train_and_validation_accuracy
import tensorflow_core as tf_core
import random as rn
from tensorflow.keras import backend as K
np.random.seed(42)
rn.seed(42)
tf_core.random.set_seed(42)

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
    """indexes_of_target_features = [features_without_date.index(f) for f in features_without_date if
                                   f.startswith('Close')]"""
    indexes_of_target_features = [features_without_date.index(f) for f in features_without_date if
                                  f.startswith('trend')]

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
            predictions_file[crypto+ "_observed_class"] = []
            predictions_file[crypto + "_predicted_class"] = []

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
            #print(train[0])
            x_train = train[:, :-1,:]

            #separate the training, by cryptocurrency and remove the last day before the day to predict, by doing -1
            # returns an array with all the values of the feature trend for each cryptocurrency
            y_trains=[]
            for index in indexes_of_target_features:
                y_trains.append(train[:, -1,  index])

            #delete the columns trend_1,trend_2 ecc..
            x_train=np.delete(x_train,indexes_of_target_features,2)

            # NOTE: in the testing set we must have the dates to evaluate the experiment without the date to forecast!!!
            # remove the day to predict
            # e.g date to predict 2019-01-07 thus the data about 2019-01-07 will be discarded.
            # e.g [[items],[items2],[items3]] becames [[items1],[items2]]
            x_test = test[:, :-1, :]
            # delete trend_1,trend_2 ecc..
            x_test = np.delete(x_test, indexes_of_target_features, 2)
            # remove the last day before the day to predict, by doing -1
            # returns an array with all the values of the feature close to predict!
            y_tests = []
            for index in indexes_of_target_features:
                y_tests.append(test[:, -1, index])

            # change the data type, from object to float
            x_train = x_train.astype('float')
            #y_train = y_train.astype('float')
            x_test = x_test.astype('float')
            #y_test = y_test.astype('float')

            #one hot encode for each y_train and y_test
            y_trains_encoded=[]
            for y_train in y_trains:
                y_trains_encoded.append(to_categorical(y_train))
            y_trains_encoded=np.asarray(y_trains_encoded)

            y_tests_encoded = []
            for y_test in y_tests:
                y_tests_encoded.append(to_categorical(y_test))
            y_tests_encoded = np.asarray(y_tests_encoded)

            BATCH_SIZE=x_train.shape[0]
            # if the date to predict is the first date in the testing_set
            #if date_to_predict == testing_set[0]:
            model, history = train_multi_target_model(x_train, y_trains_encoded,
                                         num_neurons=num_neurons,
                                         learning_rate=learning_rate,
                                         dropout=DROPOUT,
                                         epochs=EPOCHS,
                                         batch_size=BATCH_SIZE,
                                         num_categories=len(y_trains_encoded[0][0]),
                                         date_to_predict=date_to_predict,
                                         model_path=model_path,patience=PATIENCE)
            # plot neural network's architecture
            plot_model(model, to_file=model_path + "neural_network.png", show_shapes=True,
                       show_layer_names=True, expand_nested=True, dpi=150)

            #todo generalizzare.....
            #print(history.history.keys())

            #plot total loss
            filename = "total_train_val_loss_bs_" + str(BATCH_SIZE) + "_target_" + str(date_to_predict)
            plot_train_and_validation_loss(pd.Series(history.history['loss']),
                                           pd.Series(history.history['val_loss']),
                                           model_path, filename)
            m = 0
            while m < len(y_trains_encoded):
                curr_crypto="trend_"+str(m)
                filename = curr_crypto+"_train_val_loss_bs_" + str(BATCH_SIZE) + "_target_" + str(date_to_predict)
                plot_train_and_validation_loss(pd.Series(history.history[curr_crypto+'_loss']),
                                               pd.Series(history.history['val_'+curr_crypto+'_loss']),
                                               model_path, filename)
                # plot accuracy
                filename = curr_crypto+"_train_val_accuracy_bs_" + str(BATCH_SIZE) + "_target_" + str(date_to_predict)
                plot_train_and_validation_accuracy(pd.Series(history.history[curr_crypto+'_accuracy']),
                                               pd.Series(history.history['val_'+curr_crypto+'_accuracy']), model_path,
                                               filename)
                m+=1

            # Predict for each date in the validation set
            test_prediction = model.predict(x_test)

            # this is important!!
            K.clear_session()
            tf_core.random.set_seed(42)
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

            #save the accuracies values
            i = 0 #i represents the number of current day
            for d in days:
                predictions_file['date'].append(d)
                observed_decoded = []
                predicted_decoded = []
                print("Predicting for: ", d)

                j=0 #j represents the number of cryptocurrencies
                while j < len(y_tests):
                    observed_decoded.append(np.argmax(y_tests_encoded[j][i]))
                    predicted_decoded.append(np.argmax(test_prediction[j][i]))
                    j+=1

                for crypto, observed in zip(cryptos, np.asarray(observed_decoded)):
                    predictions_file[crypto + "_observed_class"].append(observed)
                for crypto, predicted in zip(cryptos, np.asarray(predicted_decoded)):
                    predictions_file[crypto + "_predicted_class"].append(predicted)

                # just some output to show
                print("Predicted: ", predicted_decoded)
                print("Actual: ", observed_decoded)
                i+=1
            print("\n")

        #divides results by crypto.
        for crypto in cryptos:
            PATH_CRYPTO=EXPERIMENT_PATH + "/" + RESULT_PATH + "/" + crypto+"/"+configuration_name+"/"+statistics+"/"
            folder_creator(PATH_CRYPTO, 0)

            crypto_prediction_file = {}
            crypto_prediction_file['date'] = predictions_file['date']
            crypto_prediction_file['observed_class'] = predictions_file[crypto + '_observed_class']
            crypto_prediction_file['predicted_class'] = predictions_file[crypto + '_predicted_class']

            crypto_macro_avg_recall_file = {'symbol': [], 'macro_avg_recall': []}
            crypto_macro_avg_recall_file['symbol'].append(crypto)
            performances = get_classification_stats(crypto_prediction_file['observed_class'], crypto_prediction_file['predicted_class'])
            crypto_macro_avg_recall_file['macro_avg_recall'].append(performances.get('macro avg').get('recall'))

            #serialization
            pd.DataFrame(data=crypto_prediction_file).to_csv(PATH_CRYPTO + 'predictions.csv',index=False)
            pd.DataFrame(data=crypto_macro_avg_recall_file).to_csv(PATH_CRYPTO + 'macro_avg_recall.csv',index=False)

    return
