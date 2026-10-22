import joblib
import uvicorn

import pandas as pd
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = FastAPI()

with open("realty_model.pkl", 'rb') as file:
    model = joblib.load(file)


class ModelRequestData(BaseModel):
    total_square: float
    rooms: float
    floor: float
    lat: float
    lon: float
    object_type: str
    city: str


class Result(BaseModel):
    result: float


@app.get("/health")
def health():
    return JSONResponse(content={"message": "It's alive!"}, status_code=200)

@app.get("/predict_get", response_model=Result)
def predict_get(
    total_square: float = Query(...),
    rooms: float = Query(...),
    floor: float = Query(...),
    lat: float = Query(...),
    lon: float = Query(...),
    object_type: str = Query(...),
    city: str = Query(...),
):
    input_data = {
        "total_square": total_square,
        "rooms": rooms,
        "floor": floor,
        "lat": lat,
        "lon": lon,
        "object_type": object_type,
        "city": city,
    }
    input_df = pd.DataFrame(input_data, index=[0])
    result = model.predict(input_df)[0]
    return Result(result=result)

@app.post("/predict_post", response_model=Result)
def predict_post(data: ModelRequestData):
    input_data = data.dict()
    input_df = pd.DataFrame(input_data, index=[0])
    result = model.predict(input_df)[0]
    return Result(result=result)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)