# Imports
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
import pickle
import pandas as pd
from fastapi.encoders import jsonable_encoder
import os

# Threshold
thres = 0.55

app = FastAPI()

# Definir rutas


BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "Parte_A", "modelo_proyecto_final.pkl")
COLUMNS_PATH = os.path.join(BASE_DIR, "data", "categories_ohe_without_fraudulent.pickle")
BINS_ORDER = os.path.join(BASE_DIR, "data", "saved_bins_order.pickle")
BINS_TRANSACTION = os.path.join(BASE_DIR, "data", "saved_bins_transaction.pickle")

# Cargar modelo
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
# Cargar columnas
with open(COLUMNS_PATH, 'rb') as handle:
        ohe_tr = pickle.load(handle)
# Cargar bins order
with open(BINS_ORDER, 'rb') as handle:
    new_saved_bins_order = pickle.load(handle)
# Cargar bins transaction
with open(BINS_TRANSACTION, 'rb') as handle:
    new_saved_bins_transaction = pickle.load(handle)
# Using original columns, API change it to ohe --> What model needs
class Answer(BaseModel):
    orderAmount : float
    orderState : str
    paymentMethodRegistrationFailure: str
    paymentMethodType : str
    paymentMethodProvider: str
    paymentMethodIssuer : str
    transactionAmount : int
    transactionFailed : bool
    emailDomain : str
    emailProvider : str
    customerIPAddressSimplified : str
    sameCity : str


@app.get("/")
async def root():
    return {"message": "This is final project for a bootcamp EDVAI! "}


@app.post("/prediction")
def predict_fraud_customer(answer: Answer):
    answer_dict = jsonable_encoder(answer)
    
    for key, value in answer_dict.items():
        answer_dict[key] = [value]

    # Crear un dataframe
    single_instance = pd.DataFrame.from_dict(answer_dict)
    
    
    # Manejo de bins
    single_instance["orderAmount"] = single_instance["orderAmount"].astype(float)
    single_instance["orderAmount"] = pd.cut(single_instance['orderAmount'],
                                     bins=new_saved_bins_order, 
                                     include_lowest=True)
    
    single_instance["transactionAmount"] = single_instance["transactionAmount"].astype(float)
    single_instance["transactionAmount"] = pd.cut(single_instance['transactionAmount'],
                                     bins=new_saved_bins_transaction, 
                                     include_lowest=True)


    # Aplicar one hot encoding a la data entrante
    single_instance_ohe = pd.get_dummies(single_instance).reindex(columns = ohe_tr).fillna(0)
    
    probabilities = model.predict_proba(single_instance_ohe)
    probabiltity_positive_class = probabilities[0][1]

    # Actualizar el score en función del threshold definido
    score = 1 if probabiltity_positive_class >= thres else 0

    response = {"score": score}
    return response


if __name__ == '__main__':
    # Utilizo 127.0.0.1 porque 0.0.0.0 no funciona en mi máquina
    uvicorn.run(app, host='127.0.0.1', port=8000)

