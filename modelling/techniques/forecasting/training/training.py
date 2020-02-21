import numpy as np
import pandas as pd

from utility.dataset_utils import cut_dataset_by_range

"""from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import LSTM, Dropout, Dense
from keras.optimizers import Adam"""
from sklearn.preprocessing import MinMaxScaler


def prepare_input_forecasting(PREPROCESSED_PATH,CLUSTERING_CRYPTO_PATH,crypto):
    # il dataset è quello che hai letto.
    # le features sono quelle che sono inclusa la data
    # le features senza data le ottieni in due secondi
    # la target feature scalata la ottieni facilmente, basta estrarla facendo solo "fit".
    df = pd.read_csv(CLUSTERING_CRYPTO_PATH+crypto, sep=',',header=0)
    df1 = pd.read_csv(PREPROCESSED_PATH + crypto, sep=',', header=0)
    #cut_dataset_by_range(PATH_SOURCE, crypto.replace(".csv",""), start_date, end_date)
    #select only the valid features
    #valid_features = [feature for feature in df.columns if feature not in features_to_exclude]
    features_without_date = [feature for feature in df.columns if feature!="Date"]

    scaler_target_feature = MinMaxScaler()
    #scaling solo della colonna "close"
    #scaler_target_feature.fit(df.loc[:, [col for col in df.columns if col.startswith('Close')]])
    scaled=scaler_target_feature.fit_transform(df1["Close"].values.reshape(-1,1))
    df1['Close']=scaled.reshape(-1)
    # scaler di tutte le features tranne "date"
    #df.loc[:, df.columns != 'Date'] = scaler.fit_transform(df.loc[:, df.columns != 'Date'])
    return df,df.columns,features_without_date, df1


def fromtemporal_totensor(dataset, window_considered, output_path, output_name):
    try:
        #pickling is also known as Serialization
        #The pickle module is not secure. Only unpickle data you trust.
        #load is for de-serialize
        #allow_pickle=True else: Object arrays cannot be loaded when allow_pickle=False
        file_path=output_path + "/crypto_TensorFormat_" + output_name + "_" + str(window_considered) + '.npy'
        lstm_tensor = np.load(file_path,allow_pickle=True)
        print('LSTM Version found!')
        return lstm_tensor
    except FileNotFoundError as e:
        print('LSTM version not found. Creating..')
        # an array in this format: [[array1,array2,array3,ecc..]]
        # -num of rows: window_considered
        # -num of columns: "dataset.shape[1]"
        lstm_tensor = np.zeros((1, window_considered, dataset.shape[1]))
        # for i between 0 to (num of elements in original array - window + 1)
        for i in range(dataset.shape[0] - window_considered + 1):
            #note (i:i + window_considered) is the rows selection.
            element=dataset[i:i + window_considered, :].reshape(1, window_considered, dataset.shape[1])
            lstm_tensor = np.append(lstm_tensor, element,axis=0)#axis 0 in order to appen on rows
        #easy explanation:
            """i:0-701 (730-30+1)
               i=0; => from day 0 + 30 days 
               i=1 => from day 1 + 30 days 
                """
        #serialization
        output_path += "/crypto_"
        name_tensor = 'TensorFormat_' + output_name + '_' + str(window_considered)
        #since the first element is zero I'll skip it:
        lstm_tensor=lstm_tensor[1:,:]
        np.save(str(output_path + name_tensor),lstm_tensor)
        return lstm_tensor


def get_training_testing_set(dataset_tensor_format, date_to_predict):
    train = []
    test = []

    index_feature_date = 0
    for sample in dataset_tensor_format:
        # Candidate is a date: 2018-01-30, for example.
        # -1 is used in order to reverse the list.
        candidate = sample[-1,index_feature_date]
        candidate = pd.to_datetime(candidate)

        #if the candidate date is equal to the date to predict then it will be in test set.
        if candidate == pd.to_datetime(date_to_predict):
            test.append(sample)
        #if the candidate date is after the date to predict then ignore it.
        elif candidate > pd.to_datetime(date_to_predict):
            pass
        #otherwise,it will be in the training set
        else:
            train.append(sample)
    return np.array(train), np.array(test)


def train_model(x_train, y_train, x_test, y_test, lstm_neurons, learning_rate, dropout, epochs, batch_size, dimension_last_layer,
                model_path='', model=None):
    callbacks = [
        # Early stopping sul train o validation set? pperche qui sara' allenato su un solo esempio di unused,
        # quindi converrebbe controllare la train_loss (loss)
        EarlyStopping(monitor='loss', patience=4),
        ModelCheckpoint(
            # lo stesso qui, modificato il monitor da val_loss a loss (?)
            monitor='loss', save_best_only=True,
            filepath=model_path + 'lstm_neur{}-do{}-ep{}-bs{}.h5'.format(
                lstm_neurons, dropout, epochs, batch_size))
    ]
    if model is None:
        model = Sequential()
        model.add(LSTM(lstm_neurons, input_shape=(x_train.shape[1], x_train.shape[2])))
        model.add(Dropout(dropout))
        model.add(Dense(dimension_last_layer))
        adam=Adam(lr=learning_rate)
        model.compile(loss='mean_squared_error', optimizer=adam, metrics=['acc', 'mae'])

    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test),
                        verbose=0, shuffle=False, callbacks=callbacks)
    return model, history
