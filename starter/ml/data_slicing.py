import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle

from data import process_data
from model import inference, compute_model_metrics


def check_slice_stats(df, feature):
    for cls in df["salary"].unique():
        df_temp = df[df["salary"] == cls].copy()

        # convert categorial data to numeric data
        unique_arr = df_temp[feature].unique()
        df_temp[feature].replace(
            unique_arr, np.arange(
                len(unique_arr)), inplace=True)

        mean = df_temp[feature].mean()
        std = df_temp[feature].std()
        print(f"Salary type: {cls}")
        print(f"- {feature} mean: {mean:.4f}")
        print(f"- {feature} stddev: {std:.4f}")
        print("\n")


def check_performance_with_slice(df, feature):
    _, test = train_test_split(df, test_size=0.2)

    with open("model/knn_model.pkl", 'rb') as f:
        model = pickle.load(f)

    with open("model/encoder.pkl", 'rb') as f:
        encoder = pickle.load(f)

    with open("model/lb.pkl", 'rb') as f:
        lb = pickle.load(f)

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    with open("output/slice_output.txt", "w") as f:
        for cls in test[feature].unique():
            df_slice = test[test[feature] == cls]

            # encode data
            X_test, y_test, _, _ = process_data(
                X=df_slice,
                categorical_features=cat_features,
                label="salary",
                encoder=encoder,
                lb=lb,
                training=False)

            # inference
            preds = inference(model, X_test)

            precision, recall, f1 = compute_model_metrics(y_test, preds)

            f.write(f'Feature: "{feature}" - <{cls}> \n')
            f.write(f'>> Precision: {precision} \n')
            f.write(f'>> Recall: {recall} \n')
            f.write(f'>> F1: {f1} \n')


if __name__ == '__main__':
    path = "./data/census.csv"
    data = pd.read_csv(path, skipinitialspace=True)

    # check_slice_stats(data, "education")
    # check_slice_stats(data, "workclass")
    # check_slice_stats(data, "marital-status")
    # check_slice_stats(data, "occupation")

    check_performance_with_slice(data, "education")
