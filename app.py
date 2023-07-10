from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import pickle
import pandas as pd

from starter.ml.model import inference
from starter.ml.data import process_data

# the data object with user information


class UserInfo(BaseModel):
    age: int
    workclass: Optional[str]
    fnlgt: int
    education: Optional[str]
    education_num: int
    marital_status: Optional[str]
    occupation: Optional[str]
    relationship: Optional[str]
    race: Optional[str]
    sex: Optional[str]
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: Optional[str]


# instantiate the app
app = FastAPI()

# GET on the root giving a welcome message


@app.get("/")
async def say_hello():
    return {"greeting": "Hello World!"}

# POST that does model inference


@app.post("/predict/")
async def predict(data: UserInfo):
    # convert data to dataframe
    data = vars(data)
    df = pd.DataFrame(data, index=[0])

    # load pretrained model and encoder
    with open("model/knn_model.pkl", 'rb') as f:
        model = pickle.load(f)

    with open("model/encoder.pkl", 'rb') as f:
        encoder = pickle.load(f)

    with open("model/lb.pkl", 'rb') as f:
        lb = pickle.load(f)

    # encode data
    cat_features = [
        "workclass",
        "education",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native_country",
    ]
    X_test, _, _, _ = process_data(
        X=df,
        categorical_features=cat_features,
        encoder=encoder,
        lb=lb,
        training=False)

    # inference
    pred = inference(model, X_test)
    if pred[0]:
        return {"Salary": ">50K"}
    else:
        return {"Salary": "<=50K"}
