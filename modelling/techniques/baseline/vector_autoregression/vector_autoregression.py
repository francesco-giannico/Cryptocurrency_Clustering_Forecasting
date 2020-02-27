import math
from datetime import datetime as dt
from datetime import timedelta
import numpy as np
import pandas as pd
import os
from statsmodels.tsa.api import VAR
import warnings

from utility.folder_creator import folder_creator

warnings.filterwarnings("ignore")


# # Utility functions
def str_to_datetime(inp_dt):
    # return dt.strptime(inp_dt, '%Y-%m-%d')
    return dt.strptime(inp_dt, '%d/%m/%y')


def str_to_datetime2(inp_dt):
    return dt.strptime(inp_dt, '%Y-%m-%d')


def datetime_to_str(inp_dt):
    return inp_dt.strftime('%Y-%m-%d')
    # return inp_dt.strftime('%d/%m/%y')


def datetime_to_str2(inp_dt):
    return inp_dt.strftime('%d/%m/%y')


def scale_data(old_val, old_min, old_max, new_min, new_max):
    return float((((old_val - old_min) / (old_max - old_min)) * (new_max - new_min)) + new_min)


# def check_date_exist(dt_to_check):
#     return True if dt_to_check in timeframe_list_dt else False
def check_date_exist(dt_to_check):
    return True  # modificare becera che non controlla la data da testare

result_folder="../modelling/techniques/baseline/simple_prediction/output/"
partial_folder="predictions"
final_folder="average_rmse"


#PER IL MULTITARGET!!!!
def vector_autoregression(data_path,test_set):
    folder_creator(result_folder+partial_folder+"/",1)
    folder_creator(result_folder+final_folder,1)

    csv="../../crypto_preprocessing/step5_horizontal/horizontal.csv"

    stock_df = pd.read_csv(csv, sep=',', decimal='.', header=0)
    stock_df['DateTime'] = stock_df['DateTime'].apply(lambda x: str_to_datetime2(x))
    stock_df.set_index('DateTime', inplace=True)
    stock_df.sort_index(inplace=True)

    print(stock_df.columns)
    print(stock_df.index)
    features = stock_df.columns
    features = [f for f in features if f.startswith('Close')]
    new_model = stock_df[features]
    new_model = new_model.apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    print(new_model.head())
    print('features:', len(features))
    print('sono:', features)

    for date in test_set:

        try:
            test_tf = str_to_datetime2(date_pred)
            train_tf = test_tf - timedelta(days=1)

            if not check_date_exist(train_tf):
                train_tf = train_tf - timedelta(days=1)

                if not check_date_exist(train_tf):
                    train_tf = train_tf - timedelta(days=1)

                    if not check_date_exist(train_tf):
                        print('Error while trying yo get train/test timeframe.')
                        continue

            print('Last Train day: {}'.format(datetime_to_str(train_tf)))
            print('Test day: {}'.format(date_pred))

            train_model = new_model[:train_tf]
            train_model.fillna(0, inplace=True)
            y_test = new_model[test_tf:test_tf].values[0]

            model = VAR(train_model)
            results = model.fit(maxlags=3, ic='aic')
            lag_order = results.k_ar
            y_predicted = results.forecast(train_model.values[-lag_order:], 1)[0]

            os.makedirs(out_path, exist_ok=True)
            with open(os.path.join(out_path,partial_folder,'var_model_norm_{}.csv'.format(date_pred)), 'w') as vf:
                vf.write('Real,Predicted\n')
                for k in range(len(y_test)):
                    vf.write('{},{}\n'.format(y_test[k], y_predicted[k]))


        except Exception as e:
            print('Error, possible cause: {}'.format(e))


    errors=[]

    for csv in os.listdir(os.path.join(out_path,partial_folder)):
        res = pd.read_csv(os.path.join(out_path,partial_folder,csv))
        error = res['Real'] - res['Predicted']
        sq_error = error ** 2
        errors.append(np.mean(sq_error))

    with open(os.path.join(out_path,final_folder,"RMSE.txt"), 'w+') as out:
        final = math.sqrt(np.mean(errors))
        out.write(str(final))