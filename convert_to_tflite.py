import tensorflow as tf

# Cargar modelo .h5
model = tf.keras.models.load_model("models/food_model.h5",
    compile=False)

# Convertidor a TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Optimización (MUY IMPORTANTE)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convertir
tflite_model = converter.convert()

# Guardar archivo
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

print("Modelo convertido a TFLite correctamente")