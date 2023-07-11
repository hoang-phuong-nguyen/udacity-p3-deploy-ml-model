from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Optional
import pickle
import pandas as pd

from starter.ml.model import inference
from starter.ml.data import process_data

# the data object with user information


class UserInfo(BaseModel):
    age: int = Field(..., example=31)
    workclass: Optional[str] = Field(..., example="Private")
    fnlgt: int = Field(..., example=45781)
    education: Optional[str] = Field(..., example="Masters")
    education_num: int = Field(..., example=14)
    marital_status: Optional[str] = Field(..., example="Never-married")
    occupation: Optional[str] = Field(..., example="Prof-specialty")
    relationship: Optional[str] = Field(..., example="Not-in-family")
    race: Optional[str] = Field(..., example="White")
    sex: Optional[str] = Field(..., example="Female")
    capital_gain: int = Field(..., example=14084)
    capital_loss: int = Field(..., example=0)
    hours_per_week: int = Field(..., example=50)
    native_country: Optional[str] = Field(..., example="United-States")


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
