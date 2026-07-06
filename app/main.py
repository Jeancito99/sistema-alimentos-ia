from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

# import tensorflow as tf
# import numpy as np
# from PIL import Image
# import joblib
# import io
# app = FastAPI()
# @app.post("/")
# def home():
#     return {
#         "status": "ok",
#         "api": "Sistema de Predicción de Vida Útil"
#     }
# app.add_middleware(
#     CORSMiddleware,
#     # allow_origins=["https://sistema-alimentos-laravel.onrender.com"],  # En producción reemplazar por el dominio de Laravel
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Variables globales
# model = None
# scaler = None


# # Cargar modelo una sola vez
# def get_model():
#     global model
#     if model is None:
#         model = tf.keras.models.load_model("models/food_model.h5",compile=False)
#     return model


# # Cargar scaler una sola vez
# def get_scaler():
#     global scaler
#     if scaler is None:
#         scaler = joblib.load("models/scaler.pkl")
#     return scaler


# # Procesar imagen
# def preprocess_image(image_bytes):
#     img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
#     img = img.resize((128, 128))
#     img_array = np.array(img, dtype=np.float32) / 255.0
#     return np.expand_dims(img_array, axis=0)


# @app.post("/predict")
# async def predict(
#     file: UploadFile = File(...),
#     temp: float = Form(...),
#     hum: float = Form(...)
# ):
#     # Obtener modelo y scaler
#     modelo = get_model()
#     scaler = get_scaler()

#     # Procesar imagen
#     img_data = preprocess_image(await file.read())

#     # Escalar datos IoT
#     iot_data = scaler.transform([[temp, hum]])

#     # Predicción
#     pred = modelo.predict([img_data, iot_data], verbose=0)

#     dias = float(pred[0][0])
#     dias= 0 if dias <= 0 else dias
    
#     return {
#         "dias_restantes": dias,
#         "estado": "Consumible" if dias > 2 else "Desechar"
#     }

app = FastAPI()
@app.post("/test")
async def test(nombre: str = Form(...)):
    return {
        "mensaje": f"Hola {nombre}",
        "recibido": nombre
    }

# Correr con: uvicorn app.main:app --reload