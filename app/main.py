from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import os

app = Flask(__name__)

# 1. Cargar el modelo .h5
MODEL_PATH = 'models/food_model.h5'
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Modelo model.h5 cargado exitosamente.")
else:
    model = None
    print(f"¡Alerta! No se encontró el archivo '{MODEL_PATH}' en el directorio actual.")

# Función para preprocesar la imagen (ajusta el tamaño según tu modelo)
def preprocess_image(file_stream, target_size=(224, 224)):
    # Leer la imagen desde el buffer de memoria
    np_img = np.frombuffer(file_stream.read(), np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    
    # Redimensionar y normalizar (0 a 1)
    img = cv2.resize(img, target_size)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)  # Añadir dimensión de batch (1, H, W, C)
    return img

@app.route('/api/predict', methods=['POST'])
def predict_food_status():
    # Validar que la imagen venga en la petición
    if 'imagen' not in request.files:
        return jsonify({"error": "Falta el archivo de imagen en la petición"}), 400
    
    file = request.files['imagen']
    
    # Obtener los datos numéricos enviados desde el formulario (Laravel)
    try:
        humedad = float(request.form.get('humedad'))
        temperatura = float(request.form.get('temperatura'))
    except (TypeError, ValueError):
        return jsonify({"error": "Los parámetros 'humedad' y 'temperatura' son obligatorios y deben ser numéricos"}), 400

    if file.filename == '':
        return jsonify({"error": "No se seleccionó ningún archivo"}), 400

    try:
        # Preprocesar la imagen recibida
        processed_img = preprocess_image(file)

        # Preparar los datos numéricos como un arreglo de NumPy
        numeric_features = np.array([[humedad, temperatura]], dtype=np.float32)

        # 2. Realizar la predicción
        if model is not None:
            # NOTA: Este flujo asume que tu model.h5 acepta múltiples entradas: [imagen, datos_numericos]
            # Si tu modelo solo acepta la imagen, cambia esto a: model.predict(processed_img)
            prediction = model.predict([processed_img, numeric_features])
            
            # Asumiendo que el modelo devuelve un valor continuo (regresión) para los días
            dias = int(np.round(prediction[0][0]))
        else:
            # Caso de prueba por si ejecutas la API sin el modelo real
            dias = 3  

        # Evitar días negativos por ruido del modelo
        dias = max(0, dias)

        # 3. Formatear la respuesta JSON tal como la solicitaste
        response_data = {
            "dias_restantes": dias,
            "estado": "Consumible" if dias > 2 else "Desechar"
        }

        return jsonify(response_data), 200

    except Exception as e:
        return jsonify({"error": f"Error interno en la predicción: {str(e)}"}), 500

if __name__ == '__main__':
    # Ejecuta la API en el puerto 5000
    app.run(host='0.0.0.0', port=5000, debug=True)