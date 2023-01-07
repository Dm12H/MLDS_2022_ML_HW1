import pandas as pd
from fastapi import FastAPI, Depends, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List
import pickle
from io import StringIO
import json

from model_utils import PredictionModel, CarEncoder


MODEL_PATH = "model.pickle"
app = FastAPI()
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)


class Estimator:
    def __init__(self, prebuilt_model):
        self._inner_model = prebuilt_model

    def predict_single(self, item):
        df = pd.DataFrame(columns=self._inner_model.columns)
        df = df.append(json.load(StringIO(item.json())), ignore_index=True)
        prediction, = self._inner_model.predict(df)
        return int(prediction)

    def predict_multiple(self, items):
        df = pd.DataFrame(columns=self._inner_model.columns)
        df.append(json.load(StringIO(items.json())), ignore_index=True)
        prediction = self._inner_model.predict(df)
        return [int(p) for p in prediction]

    def predict_df(self, df):
        prediction = self._inner_model.predict(df)
        return prediction


def get_estimator():
    return Estimator(prebuilt_model=model)


class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]


class Prediction(BaseModel):
    selling_price: int


class Predictions(BaseModel):
    objects: List[Prediction]


@app.post("/predict_item")
def predict_item(item: Item, estimator: Estimator = Depends(get_estimator)) -> Prediction:
    """
    Сделать предсказание для 1 объекта в виде json
    """
    rval = estimator.predict_single(item)
    return Prediction(selling_price=rval)


@app.post("/predict_items")
def predict_items(items: List[Item], estimator: Estimator = Depends(get_estimator)) -> Predictions:
    """
    сделать предсказание для списка объектов в формате json
    """
    rlist = estimator.predict_multiple(items)
    predictions = [Prediction(selling_price=val) for val in rlist]
    return Predictions(objects=predictions)


@app.post("/predict_items_csv", response_class=StreamingResponse)
def predict_items_csv(csv: UploadFile, estimator: Estimator = Depends(get_estimator)):
    """
    сделать предсказание для файла csv.
    мы предполагаем, что файл правильного формата и со всеми нужными колонками,
    selling_price при этом в запросе отсутствует
    """
    df = pd.read_csv(csv)
    prediction = estimator.predict_df(df)
    df.insert(loc=2, column="selling_price", value=prediction)
    response_buf = StringIO()
    df.to_csv(response_buf, index=False)
    # формирование response взято отсюда: https://stackoverflow.com/a/69799463
    response = StreamingResponse(
        iter([response_buf.getvalue()]),
        media_type='text/csv',
        headers={
            'Content-Disposition': 'attachment;filename=dataset.csv',
            'Access-Control-Expose-Headers': 'Content-Disposition'
        }
    )
    return response


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app")