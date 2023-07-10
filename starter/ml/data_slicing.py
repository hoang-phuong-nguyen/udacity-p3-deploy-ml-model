import pandas as pd
import numpy as np

def data_slice(df, feature):
    for cls in df["salary"].unique():
        df_temp = df[df["salary"]==cls].copy()
        
        # convert categorial data to numeric data
        unique_arr = df_temp[feature].unique()
        df_temp[feature].replace(unique_arr, np.arange(len(unique_arr)), inplace=True)
    
        mean = df_temp[feature].mean()
        std = df_temp[feature].std()
        print(f"Salary type: {cls}")
        print(f"- {feature} mean: {mean:.4f}")
        print(f"- {feature} stddev: {std:.4f}")
        print("\n")
    
if __name__ == '__main__':
    path = "./data/census.csv"
    data = pd.read_csv(path, skipinitialspace=True)
    
    data_slice(data, "education")
    data_slice(data, "workclass")
    data_slice(data, "marital-status")
    data_slice(data, "occupation")
    
    
    