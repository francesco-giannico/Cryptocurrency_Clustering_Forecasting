from sklearn.preprocessing import MinMaxScaler
import pandas as pd


def normalize(filepath, features_to_exclude, output_path, filename_output):
    data = pd.read_csv(filepath, sep=',')
    scaler = MinMaxScaler()
    for col in data.columns:
        if col not in features_to_exclude:
            normalized = scaler.fit_transform(data[col].values.reshape(-1, 1))
            data[col] = pd.Series(normalized.reshape(-1))
    data.to_csv(output_path + filename_output + "_normalized.csv", index=False)
    return