import os
import pandas as pd
from scipy.stats import boxcox

from utility.folder_creator import folder_creator


def power_transformation_1(input_path,output_path):
    folder_creator(output_path,1)
    for type_of_normalization in os.listdir(input_path):
        #todo remove this
        if type_of_normalization=="min_max_normalized":
            for crypto in os.listdir(input_path+type_of_normalization):
                df = pd.read_csv(input_path +type_of_normalization+"/"+ crypto, sep=",", header=0)
                for feature in df.columns.values:
                    if feature!="Date":
                        df[feature] = boxcox(df[feature]+0.0000001,0.0)
                df.to_csv(output_path + crypto, sep=",", index=False)

def power_transformation(input_path,output_path):
    folder_creator(output_path,1)

    for crypto in os.listdir(input_path):
        df = pd.read_csv(input_path+ crypto, sep=",", header=0)
        for feature in df.columns.values:
            if feature!="Date":
                df[feature],lam= boxcox(df[feature]+0.0000001)
                print('Feature: '+ feature + '\nLambda: %f' % lam)
        df.to_csv(output_path + crypto, sep=",", index=False)