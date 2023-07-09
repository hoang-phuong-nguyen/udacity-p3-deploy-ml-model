import os
import pandas as pd
import pytest

from .data import clean_data

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
