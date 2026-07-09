from fastapi import FastAPI, File, UploadFile, Form
import tensorflow as tf
import numpy as np
import cv2
import os

app = FastAPI(title="Food API Adaptada")

# Intentar cargar el modelo .h5 al iniciar
MODEL_PATH = 'model.h5'
model = None
model_error_msg = None

try:
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Modelo model.h5 cargado exitosamente.")
    else:
        model_error_msg = "Archivo 'model.h5' no encontrado en el servidor."
except Exception as e:
    model_error_msg = f"Error al cargar el archivo .h5: {str(e)}"
    print(model_error_msg)


def preprocess_image(file_bytes, target_size=(224, 224)):
    np_img = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("El archivo enviado no pudo ser decodificado como una imagen válida.")
    
    img = cv2.resize(img, target_size)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img


# Ruta cambiada a /predict para coincidir con tu Laravel
@app.post("/predict")
async def predict_food_status(
    file: UploadFile = File(...),  # Cambiado de 'imagen' a 'file'
    temp: float = Form(...),       # Cambiado de 'temperatura' a 'temp'
    hum: float = Form(...)         # Cambiado de 'humedad' a 'hum'
):
    # Estructura base de la respuesta JSON
    response_data = {
        "success": False,
        "dias_restantes": 0,
        "estado": "Desechar",
        "error": None
    }

    try:
        # 1. Verificar si el modelo cargó correctamente al inicio
        if model is None:
            response_data["error"] = f"El modelo de IA no está disponible. Motivo: {model_error_msg}"
            # Contingencia: predicción genérica basada en reglas básicas
            response_data["dias_restantes"] = 3 if temp < 25 else 1
            response_data["estado"] = "Consumible" if response_data["dias_restantes"] > 2 else "Desechar"
            return response_data

        # 2. Procesamiento de la imagen usando el parámetro 'file'
        file_bytes = await file.read()
        processed_img = preprocess_image(file_bytes)
        numeric_features = np.array([[hum, temp]], dtype=np.float32)

        # 3. Predicción
        prediction = model.predict([processed_img, numeric_features])
        
        # Procesar resultado del modelo
        dias = int(np.round(prediction[0][0]))
        dias = max(0, dias)  # Evitar números negativos

        # 4. Respuesta exitosa en limpio
        response_data["success"] = True
        response_data["dias_restantes"] = dias
        response_data["estado"] = "Consumible" if dias > 2 else "Desechar"
        return response_data

    except Exception as e:
        # Captura cualquier error en tiempo de ejecución
        response_data["success"] = False
        response_data["error"] = f"Error en el procesamiento interno de la API: {str(e)}"
        response_data["dias_restantes"] = 0
        response_data["estado"] = "Desechar"
        return response_data