import os
from datetime import timedelta
import pandas as pd
import numpy as np
from modelling.techniques.forecasting.evaluation.error_measures import get_rmse, get_accuracy, get_classification_stats
from utility.folder_creator import folder_creator


partial_folder="predictions"
folder_rmse="average_rmse"
folder_accuracy="average_accuracy"

"""def simple_prediction(data_path,test_set,result_folder):
    folder_creator(result_folder+partial_folder+"/",1)
    folder_creator(result_folder+folder_rmse,1)
    folder_creator(result_folder + folder_accuracy, 1)
    for crypto in os.listdir(data_path):
        df= pd.read_csv(data_path+crypto,usecols=['Date','Close','trend'])

        #new dataframe for output
        df1=pd.DataFrame(columns=["date","observed_value","predicted_value","observed_class","predicted_class"])
        for date_to_predict in test_set:
            day_before = (pd.to_datetime(date_to_predict,format="%Y-%m-%d") - timedelta(days=1)).strftime('%Y-%m-%d')

            row_day_before=df[df['Date']==day_before]
            row_day_before = row_day_before.set_index('Date')

            row_day_to_predict=df[df['Date']==date_to_predict]
            row_day_to_predict=row_day_to_predict.set_index('Date')

            df1 = df1.append({'date':date_to_predict,'observed_value':row_day_to_predict.loc[date_to_predict,'Close'],
                              'predicted_value':row_day_before.loc[day_before,'Close'],
                              'observed_class': row_day_to_predict.loc[date_to_predict, 'trend'],
                              'predicted_class': row_day_before.loc[day_before, 'trend'],
                              },ignore_index=True)
        df1.to_csv(result_folder+partial_folder+"/"+crypto,sep=",",index=False)

    #accuracy and rmse
    rmses=[]
    accuracies=[]
    for crypto in os.listdir(result_folder+partial_folder+"/"):
        df = pd.read_csv(result_folder+partial_folder+"/"+crypto)
        #get rmse for each crypto
        rmse = get_rmse(df['observed_value'], df['predicted_value'])
        accuracy= get_accuracy(df['observed_class'], df['predicted_class'])
        rmses.append(rmse)
        accuracies.append(accuracy)
        with open(os.path.join(result_folder,folder_rmse, crypto.replace(".csv","")), 'w+') as out:
            out.write(str(rmse))
        with open(os.path.join(result_folder,folder_accuracy, crypto.replace(".csv","_accuracy.txt")), 'w+') as out:
            out.write(str(accuracy))

    # average accuracy
    with open(os.path.join(result_folder, folder_accuracy, "average_accuracy.txt"), 'w+') as out:
        final = np.mean(accuracies)
        out.write(str(final))"""

"""def simple_prediction(data_path,test_set,result_folder):
    folder_creator(result_folder+partial_folder+"/",1)
    folder_creator(result_folder+folder_rmse,1)
    folder_creator(result_folder + folder_accuracy, 1)
    for crypto in os.listdir(data_path):
        df= pd.read_csv(data_path+crypto,usecols=['Date','Close','trend'])
        #new dataframe for output
        df1=pd.DataFrame(columns=["date","observed_class","predicted_class"])
        final_class=None
        for date_to_predict in test_set:
            day_before = (pd.to_datetime(date_to_predict,format="%Y-%m-%d") - timedelta(days=1)).strftime('%Y-%m-%d')

            row_day_before=df[df['Date']==day_before]
            row_day_before = row_day_before.set_index('Date')
            if final_class==None:
                final_class=row_day_before.loc[day_before, 'trend']

            row_day_to_predict=df[df['Date']==date_to_predict]
            row_day_to_predict=row_day_to_predict.set_index('Date')

            df1 = df1.append({'date':date_to_predict,
                              'observed_class': row_day_to_predict.loc[date_to_predict, 'trend'],
                              'predicted_class': final_class,
                              },ignore_index=True)
        df1.to_csv(result_folder+partial_folder+"/"+crypto,sep=",",index=False)
    #accuracy and rmse
    #rmses=[]
    accuracies=[]
    for crypto in os.listdir(result_folder+partial_folder+"/"):
        df = pd.read_csv(result_folder+partial_folder+"/"+crypto)
        #get rmse for each crypto
        performances= get_classification_stats(df['observed_class'], df['predicted_class'])
        #rmses.append(rmse)
        accuracies.append(performances.get('macro avg').get('recall'))
        with open(os.path.join(result_folder,folder_accuracy, crypto.replace(".csv","_accuracy.txt")), 'w+') as out:
            out.write(str(performances.get('macro avg').get('recall')))

    # average accuracy
    with open(os.path.join(result_folder, folder_accuracy, "average_accuracy.txt"), 'w+') as out:
        final = np.mean(accuracies)
        out.write(str(final))"""

def simple_prediction(data_path,test_set,result_folder,lookback_days):
    folder_creator(result_folder+partial_folder+"/",1)
    folder_creator(result_folder+folder_rmse,1)
    folder_creator(result_folder + folder_accuracy, 1)
    for crypto in os.listdir(data_path):
        df= pd.read_csv(data_path+crypto,usecols=['Date','trend'])
        #new dataframe for output
        df1=pd.DataFrame(columns=["date","observed_class","predicted_class"])
        for date_to_predict in test_set:
            n_day_before = (pd.to_datetime(date_to_predict,format="%Y-%m-%d") - timedelta(days=lookback_days)).strftime('%Y-%m-%d')

            row_day_before=df[df['Date']==n_day_before]
            row_day_before = row_day_before.set_index('Date')

            row_day_to_predict = df[df['Date'] == date_to_predict]
            row_day_to_predict = row_day_to_predict.set_index('Date')

            df1 = df1.append(
                {'date': date_to_predict, 'observed_class': row_day_to_predict.loc[date_to_predict, 'trend'],
                 'predicted_class': row_day_before.loc[n_day_before, 'trend'],
                 }, ignore_index=True)
        df1.to_csv(result_folder+partial_folder+"/"+crypto,sep=",",index=False)
    #accuracy and rmse
    #rmses=[]
    accuracies=[]
    for crypto in os.listdir(result_folder+partial_folder+"/"):
        df = pd.read_csv(result_folder+partial_folder+"/"+crypto)
        #get rmse for each crypto
        performances= get_classification_stats(df['observed_class'], df['predicted_class'])
        #rmses.append(rmse)
        accuracies.append(performances.get('macro avg').get('recall'))
        with open(os.path.join(result_folder,folder_accuracy, crypto.replace(".csv","_accuracy.txt")), 'w+') as out:
            out.write(str(performances.get('macro avg').get('recall')))

    # average accuracy
    with open(os.path.join(result_folder, folder_accuracy, "average_accuracy.txt"), 'w+') as out:
        final = np.mean(accuracies)
        out.write(str(final))