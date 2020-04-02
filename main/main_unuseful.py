def report_multi(cluster_n,folder,crypto_name):
    OUTPUT_SIMPLE_PREDICTION = "../modelling/techniques/baseline/simple_prediction/output/"
    OUTPUT_VECTOR_AUTOREGRESSION="../modelling/techniques/baseline/vector_autoregression/output/"+cluster_n

    output_name_multi="out_multi_"+crypto_name
    output_name_single = "out_single_" + crypto_name
    # baseline wins on the following cryptocurrencies:
    #folder_creator("../modelling/techniques/forecasting/comparisons/multi_target/", 0)
    gen_path_multi = "../modelling/techniques/forecasting/"+output_name_multi +"/"
    gen_path_single = "../modelling/techniques/forecasting/" + output_name_single + "/"
    for folder in os.listdir(gen_path_multi):
        for subfold in os.listdir(gen_path_multi + folder + "/multi_target/clusters/"+cluster_n+"/result/"):
            if subfold==crypto_name:
                for subfold2 in os.listdir(gen_path_multi + folder + "/multi_target/clusters/"+cluster_n+"/result/" + subfold + "/"):
                    df1 = pd.read_csv(
                        gen_path_multi + folder + "/multi_target/clusters/"+cluster_n+"/result/" + subfold + "/" + subfold2 + "/stats/errors.csv",
                        usecols=['rmse_norm'])


                    file = open(OUTPUT_SIMPLE_PREDICTION + "average_rmse/" + subfold, "r")
                    simple_model = float(file.read())
                    file.close()

                    file = open(OUTPUT_VECTOR_AUTOREGRESSION + "/rmse/" + subfold, "r")
                    vector_autoregression = float(file.read())
                    file.close()
                    best_single = 1000
                    for folder in os.listdir(gen_path_single):
                        for subfold in os.listdir(gen_path_single + folder + "/single_target/result/"):
                            if subfold == crypto_name:
                                for subfold2 in os.listdir(gen_path_single+ folder + "/single_target/result/" + subfold + "/"):
                                    df2 = pd.read_csv(
                                        gen_path_single + folder + "/single_target/result/" + "/" + subfold + "/" + subfold2 + "/stats/errors.csv",
                                        usecols=['rmse_norm'])
                                    rmse=df2['rmse_norm'][0]
                                    if(rmse<best_single):
                                        best_single=rmse

                    print("With the following multitarget config: "+ folder)
                    print("Simple baseline: " + str(simple_model))
                    print("Single target " + str(best_single))
                    print("VAR: " + str(vector_autoregression))
                    print("Multitarget " + str(df1['rmse_norm'][0]))
                    print("\n")

def report_single(crypto_name):
    OUTPUT_SIMPLE_PREDICTION = "../modelling/techniques/baseline/simple_prediction/output/"

    output_name_single = "out_single_" + crypto_name
    # baseline wins on the following cryptocurrencies:
    # folder_creator("../modelling/techniques/forecasting/comparisons/multi_target/", 0)
    gen_path_single = "../modelling/techniques/forecasting/" + output_name_single + "/"

    file = open(OUTPUT_SIMPLE_PREDICTION + "average_rmse/" + crypto_name, "r")
    simple_model = float(file.read())
    file.close()
    print("Simple baseline: " + str(simple_model))

    min=10
    config=""
    for folder in os.listdir(gen_path_single):
        print( folder)
        for subfold in os.listdir(gen_path_single + folder + "/single_target/result/"):
            if subfold == crypto_name:
                try:
                    for subfold2 in os.listdir(
                            gen_path_single + folder + "/single_target/result/" + subfold + "/"):
                        df2 = pd.read_csv(
                            gen_path_single + folder + "/single_target/result/" + "/" + subfold + "/" + subfold2 + "/stats/errors.csv",
                            usecols=['rmse_norm'])

                        print(str(df2['rmse_norm'][0]))



                        if(df2['rmse_norm'][0]<=min):
                            min=df2['rmse_norm'][0]
                            config=folder
                except:
                    pass
    print("\n")
    print("Simple baseline: " + str(simple_model))
    print("Minimo: "+ str(min))
    print("Config: " + str(config))
    if(min<=simple_model):
            print("BATTUTO!")





def report():
    OUTPUT_SIMPLE_PREDICTION = "../modelling/techniques/baseline/simple_prediction/output/"

    file = open(OUTPUT_SIMPLE_PREDICTION + "average_rmse/average_rmse.txt", "r")
    value1 = file.read()

    # baseline wins on the following cryptocurrencies:
    folder_creator("../modelling/techniques/forecasting/comparisons/single_target/", 0)
    filename = '../modelling/techniques/forecasting/comparisons/single_target/'
    gen_path = "../modelling/techniques/forecasting/outputs/"

    for folder in os.listdir(gen_path):
        df_out = {"symbol": [], "baseline": [], "single_target": [], "is_best": [], "distance_from_bs": []}
        rmses = []
        for subfold in os.listdir(gen_path + folder + "/single_target/result/"):
            for subfold2 in os.listdir(gen_path + folder + "/single_target/result/" + subfold + "/"):
                df1 = pd.read_csv(
                    gen_path + folder + "/single_target/result/" + "/" + subfold + "/" + subfold2 + "/stats/errors.csv",
                    usecols=['rmse_norm'])
                file = open(OUTPUT_SIMPLE_PREDICTION + "average_rmse/" + subfold, "r")
                value_bas = float(file.read())
                df_out['symbol'].append(subfold)
                df_out['baseline'].append(value_bas)
                df_out['single_target'].append(df1['rmse_norm'][0])
                is_best = False
                if df1['rmse_norm'][0] < value_bas:
                    is_best = True
                df_out['is_best'].append(is_best)
                distance = np.abs(df1['rmse_norm'][0] - value_bas)
                df_out['distance_from_bs'].append(distance)
                rmses.append(df1['rmse_norm'][0])

            pd.DataFrame(data=df_out).to_csv(filename + folder, index=False)

        print(folder)
        print("Baseline (AVG RMSE): " + str(value1))
        print("Single Target (AVG RMSE): " + str(np.mean(rmses)))

    path = "../modelling/techniques/forecasting/comparisons/single_target/"
    path_out = "../modelling/techniques/forecasting/comparisons/"
    df_out = pd.DataFrame()
    """min=1000
    min_name="""""
    for experiment_name in os.listdir(path):
        df = pd.read_csv(path + experiment_name)
        df_out['symbol'] = df['symbol']
        df_out['baseline'] = df['baseline']
        df_out[experiment_name] = df['single_target']

    df_out.to_csv(path_out + "final.csv", index=False)