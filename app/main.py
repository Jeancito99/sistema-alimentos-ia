from fastapi import FastAPI, UploadFile, File, Form
import tensorflow as tf
import numpy as np
from PIL import Image
import io
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # En producción, cambia esto por la URL de tu sitio Laravel
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
model = None

def get_model():
    global model
    if model is None:
        model =  tf.keras.models.load_model("model.h5")
    return model

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).resize((128, 128))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    temp: float = Form(...),
    hum: float = Form(...)
):
    # Procesar imagen e inputs numéricos
    img_data = preprocess_image(await file.read())
    iot_data = np.array([[temp, hum]])
    
    # Inferencia
    pred = model.predict([img_data, iot_data])
    
    return {
        "dias_restantes": float(pred[0][0]),
        "estado": "Consumible" if pred[0][0] > 2 else "Desechar"
    }

# Correr con: uvicorn app.main:app --reload