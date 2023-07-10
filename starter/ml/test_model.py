import os
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from .data import clean_data, process_data
from .model import train_model

@pytest.fixture
def data():
    path = "./data/census.csv"
    data = pd.read_csv(path, skipinitialspace=True)
    return clean_data(data)

def test_initial_spaces(data):
    for col in data.columns:
        assert col==col.strip(), 'There are initial spaces in column names'

def test_special_characters(data):
    assert '?' not in data.values, 'There are some special characters in the dataframe'

def test_null(data):
    assert data.shape == data.dropna().shape, 'There are some NULL values in the dataframe'

def test_duplicates(data):
    assert data.shape == data.drop_duplicates(keep='first').shape, 'There are some duplicates in the dataframe'

def test_feature_extraction(data):
    train, test = train_test_split(data, test_size=0.2)

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
    
    X_test, y_test, _, _ = process_data(
        X=test,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder_train,
        lb=lb_train)
         
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)

def test_train_model(data):
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
        X=data,
        categorical_features=cat_features,
        label="salary",
        training=True)
        
    model = train_model(X_train, y_train)
    
    assert model is not None 