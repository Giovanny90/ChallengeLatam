from challenge.model import DelayModel
from fastapi import FastAPI, HTTPException
import pandas as pd

app = FastAPI()
delay_model = DelayModel()

@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }

@app.post("/predict", status_code=200)
async def post_predict(data: dict) -> dict:
        # Extraer los datos de la solicitud JSON
    flights_data = data.get("flights")
    if not flights_data:
        raise HTTPException(status_code=400, detail="No se proporcionaron datos de vuelo")

    # Convertir los datos de vuelo en un DataFrame
    df = pd.DataFrame(flights_data)
    # Preprocesar los datos utilizando DelayModel
    features = delay_model.preprocess(df)    
    # Realizar predicciones utilizando DelayModel
    predictions = delay_model.predict(features)
    response= {"predict":predictions}
    return response

