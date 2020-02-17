""" for row in df.itertuples():
            #print(row.Open)
            if (math.isnan(row.Open)):
                fin_date=row.Index
                #df=df.drop(df.index[init_date:row.Index])
                df=df.query('index < @init_date or index > @fin_date')
                init_date=row.Index
    df.to_csv('../dataset/reviewed/'+file,",")   """