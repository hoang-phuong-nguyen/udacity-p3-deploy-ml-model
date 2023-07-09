import pandas as pd
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from sklearn.model_selection import train_test_split

from ml.data import clean_data, process_data
from ml.model import train_model, inference, compute_model_metrics

def load_data(path:str=None):
    df = pd.read_csv(path, skipinitialspace=True)
    return df
    
@hydra.main(version_base=None, config_path="../.", config_name="config")
def run(config: DictConfig):
    # load raw data
    data_path = to_absolute_path(config["data"]["raw"])
    raw_df = load_data(path=data_path)
    print(f'>> Load raw data from {data_path}... finished')
    
    # clean data
    clean_df = clean_data(raw_df)
    print('>> Clean data... finished')
    
    # save clean data
    clean_df.to_csv(to_absolute_path(config["data"]["clean"]), index=False)
    print(f'>> Save clean data to {to_absolute_path(config["data"]["clean"])}... finished')
    
    # split train, test data
    train, test = train_test_split(clean_df, test_size=config["model"]["test_size"])
    print(f'>> Split train test sets... finished')
    
    # select features
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
    
    # one hot enconding
    X_train, y_train, encoder_train, lb_train = process_data(
        X=train,
        categorical_features=cat_features,
        label="salary",
        training=True)
    print(f'>> One hot encoding for training set... finished')
    
    X_test, y_test, _, _ = process_data(
        X=test,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder_train,
        lb=lb_train)
    print(f'>> One hot encoding for test set... finished')
    
    # train model
    model = train_model(X_train, y_train, num_nb=config["model"]["num_nb"])
    print(f'>> Train model... finished')
    
    # predict
    y_pred = inference(model, X_test)
    print(f'>> Predict using the trained model... finished')
    
    # evaluate model 
    precision, recall, f1 = compute_model_metrics(y_test, y_pred)
    print(f'>> Evaluate the trained model')
    print(f'>> Precision: {precision}. Recall: {recall}. F1: {f1}')
    
if __name__ == '__main__':
    run()