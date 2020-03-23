import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.stats import stats

from sklearn.preprocessing import MinMaxScaler, QuantileTransformer
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from tensorflow.keras.layers import LSTM,Dropout,Dense
from tensorflow.keras.layers import LSTM,Dropout,Dense,Activation
import matplotlib.pyplot as plt
from tensorflow_core.python.keras.utils.vis_utils import plot_model
from utility.dataset_utils import cut_dataset_by_range

def get_scaler(PREPROCESSED_PATH,crypto,start_date,end_date):
    df1 = cut_dataset_by_range(PREPROCESSED_PATH, crypto.replace(".csv", ""), start_date, end_date)
    #todo non è detto che sia il minmax... va be.. tanto è per denormalizzare.
    scaler_target_features = MinMaxScaler()
    # save the parameters for the trasformation or the inverse_transformation for the feature "close"
    scaler_target_features.fit(df1.loc[:, [col for col in df1.columns if col.startswith('Close')]])
    return scaler_target_features

def get_scaler2(PREPROCESSED_PATH,crypto,start_date,end_date):
    df1 = cut_dataset_by_range(PREPROCESSED_PATH, crypto.replace(".csv", ""), start_date, end_date)
    p = -1
    n_t = 1
    while p <= 0.05:
        qt = QuantileTransformer(n_quantiles=n_t, random_state=0, output_distribution="normal")
        quanrtil = qt.fit_transform(df1.Close.values.reshape(-1, 1))
        new_values = pd.Series(quanrtil.reshape(-1))
        stat, p = stats.normaltest(new_values)
        if p > 0.05:
            df1.Close = pd.Series(new_values)
            # save the parameters for the trasformation or the inverse_transformation for the feature "close"
            qt.fit(df1.loc[:, [col for col in df1.columns if col.startswith('Close')]])
            print('num_quantiles:' + str(n_t))
        else:
            n_t += 1
    return qt

def prepare_input_forecasting(PREPROCESSED_PATH,CLUSTERING_CRYPTO_PATH,crypto,cryptos=None,features_to_use=None):
    #already normalized

    if features_to_use!=None:
        df = pd.read_csv(CLUSTERING_CRYPTO_PATH+crypto+".csv", sep=',',header=0,usecols=features_to_use)
    else:
        df = pd.read_csv(CLUSTERING_CRYPTO_PATH + crypto+".csv", sep=',', header=0)

    df=df.set_index("Date")
    start_date=df.index[0]
    end_date=df.index[len(df.index)-1]

    if cryptos!=None:
        #multitarget_case
        for crypto in cryptos:
            #todo review...
            # read not normalized
            """scaler_target_features=get_scaler(PREPROCESSED_PATH,crypto,start_date,end_date)
            scaler_target_features = get_scaler(PREPROCESSED_PATH, crypto, start_date, end_date)"""
    else:
        #single target case
        #read not normalized
        #TRANSFORMED_PATH="../preparation/preprocessed_dataset/transformed/"
        #scaler_target_features = get_scaler(TRANSFORMED_PATH, crypto, start_date, end_date)
        scaler_target_features = get_scaler(PREPROCESSED_PATH, crypto, start_date, end_date)
        #todo remove 7 :D
        #qt = get_scaler2(PREPROCESSED_PATH, crypto, start_date, end_date)

    df = df.reset_index()
    #exlude the feature "date"
    features_without_date = [feature for feature in df.columns if feature != "Date"]

    return df,df.columns,features_without_date,scaler_target_features


def fromtemporal_totensor(dataset, window_considered, output_path, output_name):
    """try:
        #pickling is also known as Serialization
        #The pickle module is not secure. Only unpickle data you trust.
        #load is for de-serialize
        #allow_pickle=True else: Object arrays cannot be loaded when allow_pickle=False
        file_path=output_path + "/crypto_TensorFormat_" + output_name + "_" + str(window_considered) + '.npy'
        lstm_tensor = np.load(file_path,allow_pickle=True)
        print('(LSTM Version found!)')
        return lstm_tensor
    except FileNotFoundError as e:"""
    print('LSTM version not found. Creating..')
    # an array in this format: [ [[items],[items]], [[items],[items]],.....]
    # -num of rows: window_considered
    # -num of columns: "dataset.shape[1]"
    # 1 is the number of elements in
    lstm_tensor = np.zeros((1, window_considered, dataset.shape[1]))
    # for i between 0 to (num of elements in original array - window + 1)
    """easy explanation through example:
         i:0-701 (730-30+1)
         i=0; => from day 0 + 30 days 
         i=1 => from day 1 + 30 days 
      """
    for i in range(dataset.shape[0] - window_considered + 1):
        #note (i:i + window_considered) is the rows selection.
        element=dataset[i:i + window_considered, :].reshape(1, window_considered, dataset.shape[1])
        lstm_tensor = np.append(lstm_tensor, element,axis=0)#axis 0 in order to appen on rows

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
        #takes the last date in the sample: 2017-01-09, 2017-01..., ... ,  2017-02-2019
        #since the last date is 2017-02-2019, then it is before the date to predict for example 2019-03-05, so this sample
        #will belong to the training set.
        candidate = sample[-1,index_feature_date]
        candidate = pd.to_datetime(candidate)

        #if the candidate date is equal to the date to predict then it will be in test set.
        #it happens just one time for each date to predict.
        #Test will be: [[items]] in which the items goes N(30,100,200) days before the date to predict.
        if candidate == pd.to_datetime(date_to_predict):
            test.append(sample)
        #if the candidate date is after the date to predict then ignore it.
        elif candidate > pd.to_datetime(date_to_predict):
            pass
        #otherwise,it will be in the training set
        else:
            train.append(sample)
    return np.array(train), np.array(test)

def train_model(x_train, y_train, x_test, y_test, num_neurons, learning_rate, dropout, epochs, batch_size,patience, dimension_last_layer,
                model_path='', model=None):
    #note: it's an incremental way to get a final model.
    #
    callbacks = [
        EarlyStopping(monitor='loss', patience=patience),
        ModelCheckpoint(
            monitor='loss', save_best_only=True,
            filepath=model_path+'lstm_neur{}-do{}-ep{}-bs{}.h5'.format(
                num_neurons, dropout, epochs, batch_size))
    ]

    if model is None:
        model = Sequential()
        # Add a LSTM layer with 128/256 internal units.
        model.add(LSTM(units=num_neurons,input_shape=(x_train.shape[1], x_train.shape[2])))
        #reduce the overfitting
        model.add(Dropout(dropout))
        """model.add(LSTM(units=num_neurons))
        # reduce the overfitting
        model.add(Dropout(dropout))"""
        #number of neurons of the last layer
        model.add(Dense(units=dimension_last_layer))
        #optimizer
        adam=Adam(learning_rate=learning_rate)
        #print(model.summary())
        #sgd=SGD(learning_rate=learning_rate)
        model.compile(loss='mean_squared_error', optimizer=adam, metrics=['mse'])

    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test),
                       verbose=0,shuffle=False,callbacks=callbacks, use_multiprocessing=True)
    return model, history
