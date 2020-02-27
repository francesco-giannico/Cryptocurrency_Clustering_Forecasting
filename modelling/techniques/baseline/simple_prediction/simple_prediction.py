import calendar
import os
from datetime import timedelta, datetime

import pandas as pd
import numpy as np
from math import sqrt

#PATH="../../crypto_preprocessing/step4_cutdata"
from modelling.techniques.forecasting.evaluation.error_measures import get_rmse
from utility.folder_creator import folder_creator

result_folder="../modelling/techniques/baseline/simple_prediction/output/"
partial_folder="predictions"
final_folder="average_rmse"

date=[]
def simple_prediction(data_path,test_set):
    folder_creator(result_folder+partial_folder+"/",1)
    folder_creator(result_folder+final_folder,1)

    for crypto in os.listdir(data_path):
        df= pd.read_csv(data_path+crypto,usecols=['Date','Close'])

        #new dataframe for output
        df1=pd.DataFrame(columns=["date","observed_value","predicted_value"])
        for date_to_predict in test_set:
            day_before = (pd.to_datetime(date_to_predict,format="%Y-%m-%d") - timedelta(days=1)).strftime('%Y-%m-%d')

            row_day_before=df[df['Date']==day_before]
            row_day_before = row_day_before.set_index('Date')

            row_day_to_predict=df[df['Date']==date_to_predict]
            row_day_to_predict=row_day_to_predict.set_index('Date')

            df1 = df1.append({'date':date_to_predict,'observed_value':row_day_to_predict.loc[date_to_predict,'Close'],'predicted_value':row_day_before.loc[day_before,'Close']},ignore_index=True)
        df1.to_csv(result_folder+partial_folder+"/"+crypto,sep=",",index=False)

    rmses=[]
    for crypto in os.listdir(result_folder+partial_folder+"/"):
        df = pd.read_csv(result_folder+partial_folder+"/"+crypto)
        #get rmse for each crypto
        rmses.append(get_rmse(df['predicted_value'],df['observed_value']))

    with open(os.path.join(result_folder,final_folder,"average_rmse.txt"), 'w+') as out:
        final = np.mean(rmses)
        out.write(str(final))

#Old method to compute RMSE
"""errors=[]
for crypto in os.listdir(result_folder+partial_folder+"/"):
    df = pd.read_csv(result_folder+partial_folder+"/"+crypto)
    error = df['predicted_value'] - df['observed_value'] #differenza tra colonne
    sq_error = error ** 2
    errors.append(sqrt(np.mean(sq_error)))

with open(os.path.join(result_folder,final_folder,"RMSE.txt"), 'w+') as out:
    final = np.mean(errors)
    out.write(str(final))"""
