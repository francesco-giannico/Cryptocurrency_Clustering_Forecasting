import os
from datetime import timedelta
import pandas as pd
import numpy as np
from modelling.techniques.forecasting.evaluation.error_measures import get_rmse, get_accuracy, get_classification_stats
from utility.folder_creator import folder_creator


partial_folder="predictions"
folder_performances="performances"

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

def simple_prediction(data_path,test_set,result_folder):
    folder_creator(result_folder+partial_folder+"/",1)
    folder_creator(result_folder + folder_performances, 1)
    accuracies = []
    for crypto_name in os.listdir(data_path):
        df1 = pd.DataFrame(columns=["date", "observed_class", "predicted_class"])
        for date_to_predict in test_set:
            dataset_name = crypto_name + "_" + date_to_predict + ".csv"
            df = pd.read_csv(os.path.join(data_path,crypto_name,dataset_name), usecols=['Date', 'trend'])
            #new dataframe for output
            n_day_before = (pd.to_datetime(date_to_predict,format="%Y-%m-%d") - timedelta(days=1)).strftime('%Y-%m-%d')

            row_day_before=df[df['Date']==n_day_before]
            row_day_before = row_day_before.set_index('Date')

            row_day_to_predict = df[df['Date'] == date_to_predict]
            row_day_to_predict = row_day_to_predict.set_index('Date')

            df1 = df1.append(
                {'date': date_to_predict, 'observed_class': row_day_to_predict.loc[date_to_predict, 'trend'],
                 'predicted_class': row_day_before.loc[n_day_before, 'trend'],
                 }, ignore_index=True)
        df1.to_csv(os.path.join(result_folder,partial_folder,crypto_name+".csv"),sep=",", index=False)

    for crypto in os.listdir(result_folder+partial_folder+"/"):
        df = pd.read_csv(result_folder+partial_folder+"/"+crypto)
        #get rmse for each crypto
        performances= get_classification_stats(df['observed_class'], df['predicted_class'])
        df_performances_1=pd.DataFrame()
        df_performances_2= pd.DataFrame()

        df_performances_2 = df_performances_2.append(
            {'macro_avg_precision': performances.get('macro avg').get('precision'),
             'macro_avg_recall': performances.get('macro avg').get('recall'),
             'macro_avg_f1': performances.get('macro avg').get('f1-score'),
             'weighted_macro_avg_precision': performances.get('weighted avg').get('precision'),
             'weighted_macro_avg_recall': performances.get('weighted avg').get('recall'),
             'weighted_macro_avg_f1': performances.get('weighted avg').get('f1-score'),
             'support': performances.get('weighted avg').get('support')
             }, ignore_index=True)

        z=0
        while z < 3:
            df_performances_1 = df_performances_1.append(
                {'class': str(z),
                 'precision': performances.get(str(z)).get('precision'),
                 'recall': performances.get(str(z)).get('recall'),
                 'f1_score': performances.get(str(z)).get('f1-score'),
                 'support': performances.get(str(z)).get('support')
                 }, ignore_index=True)

            z+=1

        df_performances_1.to_csv(os.path.join(result_folder,folder_performances, crypto.replace(".csv","_performances_part1.csv")),index=False)
        df_performances_2.to_csv(os.path.join(result_folder, folder_performances, crypto.replace(".csv", "_performances_part2.csv")), index=False)

        # 'accuracy', 'macro_avg_precision', 'macro_avg_recall', 'macro_avg_f1_score'
        accuracies.append(performances.get('macro avg').get('recall'))
        with open(os.path.join(result_folder,folder_performances, crypto.replace(".csv","_macro_avg_recall.txt")), 'w+') as out:
            out.write(str(performances.get('macro avg').get('recall')))

    # average macro avg recall
    with open(os.path.join(result_folder, folder_performances, "average_macro_avg_recall.txt"), 'w+') as out:
        final = np.mean(accuracies)
        out.write(str(final))