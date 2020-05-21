import os
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

PREPROCESSED_PATH="../preparation/preprocessed_dataset/cleaned/final/"
def multi_target(EXPERIMENT_PATH, DATA_PATH, TENSOR_DATA_PATH,window_sequences, list_num_neurons,
                    learning_rate,cryptos, features_to_use, DROPOUT, EPOCHS, PATIENCE,BATCH_SIZE):

    #################### FOLDER SETUP ####################
    MODELS_PATH = "models"
    RESULT_PATH = "result"

    # starting from the testing set
    for window, num_neurons in product(window_sequences, list_num_neurons):
        print('Current configuration: ')
        print("Window_sequence: ", window, "\t", "Neurons: ", num_neurons)

        # DICTIONARY FOR STATISTICS
        predictions_file = {'date': []}
        for crypto in cryptos:
            crypto = str(crypto)
            predictions_file[crypto + "_observed_class"] = []
            predictions_file[crypto + "_predicted_class"] = []

        for dataset_name in os.listdir(DATA_PATH):
            # format of dataset name: Crypto_DATE_TO_PREDICT.csv
            splitted = dataset_name.split("_")
            horizontal_name=str(splitted[0])
            print("Current crypto: ", horizontal_name, "\t")
            date_to_predict = str(splitted[1]).replace(".csv", "")
            # create a folder for data in tensor format
            folder_creator(TENSOR_DATA_PATH + "/" + horizontal_name, 0)
            # create a folder for models
            folder_creator(EXPERIMENT_PATH  + MODELS_PATH + "/", 0)
            folder_creator(EXPERIMENT_PATH + "/" + RESULT_PATH + "/", 0)

            dataset,features_filtred = prepare_input_forecasting(DATA_PATH,dataset_name,features_to_use)
            #takes all the target
            indexes_of_target_features = [features_filtred.index(f) for f in features_filtred if
                                          f.startswith('trend')]
            # non prende in input i neuroni questo.
            dataset_tensor_format = fromtemporal_totensor(np.array(dataset), window,
                                                          TENSOR_DATA_PATH + "/" + horizontal_name + "/",
                                                          horizontal_name+"_"+date_to_predict)
            #New folders for this configuration
            configuration_name = "LSTM_" + str(num_neurons) + "_neurons_" + str(window) + "_days"
            # Create a folder to save
            # - best model checkpoint
            # - statistics (results)
            statistics = "stats"
            model_path = EXPERIMENT_PATH + MODELS_PATH + "/" + configuration_name + "/"
            folder_creator(model_path,0)

            train, test = get_training_validation_testing_set(dataset_tensor_format, date_to_predict)

            train = train[:, :, 1:]
            test = test[:, :, 1:]

            x_train = train[:, :-1,:]
            #separates the y_trains
            y_trains=[]
            for index in indexes_of_target_features:
                y_trains.append(train[:, -1,  index])
            #delete the columns trend_1,trend_2 ecc..
            x_train=np.delete(x_train,indexes_of_target_features,2)

            x_test = test[:, :-1, :]
            y_tests = []
            for index in indexes_of_target_features:
                y_tests.append(test[:, -1, index])
            # delete trend_1,trend_2 ecc..
            x_test = np.delete(x_test, indexes_of_target_features, 2)

            # change the data type, from object to float
            x_train = x_train.astype('float')
            x_test = x_test.astype('float')

            #one hot encode for each y_train and y_test
            y_trains_encoded=[]
            for y_train in y_trains:
                y_trains_encoded.append(to_categorical(y_train))
            #y_trains_encoded=np.asarray(y_trains_encoded)

            y_tests_encoded = []
            for y_test in y_tests:
                y_tests_encoded.append(to_categorical(y_test))
            #y_tests_encoded = np.asarray(y_tests_encoded)

            if BATCH_SIZE==None:
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

            #decoding
            observed_decoded = []
            predicted_decoded = []
            j = 0  # j represents the number of cryptocurrencies
            while j < len(y_tests):
                observed_decoded.append(np.argmax(y_tests_encoded[j]))
                predicted_decoded.append(np.argmax(test_prediction[j]))
                j += 1

            #saving on dictionary
            predictions_file['date'].append(date_to_predict)
            for crypto, observed in zip(cryptos, np.asarray(observed_decoded)):
                predictions_file[crypto + "_observed_class"].append(observed)
            for crypto, predicted in zip(cryptos, np.asarray(predicted_decoded)):
                predictions_file[crypto + "_predicted_class"].append(predicted)

            # just some output to show
            print("Num of entries for training: ", x_train.shape[0])
            print("Predicting for: ", date_to_predict)
            print("Predicted: ", predicted_decoded)
            print("Actual: ", observed_decoded)

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
