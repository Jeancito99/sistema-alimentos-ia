# Usa una imagen base de Python
FROM python:3.11-slim

# Directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema necesarias para Keras/TensorFlow
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependencias de Python
COPY requirements.txt .
COPY requirements.txt .
COPY train_model.py .
COPY app/ ./app/

# 2. Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# 3. GENERAR EL MODELO (Aquí ocurre la magia)
# Esto ejecuta tu script y creará la carpeta /models y el archivo .h5 dentro de la imagen
RUN python train_model.py

# 4. Copiar el resto del código si es necesario
COPY . .

# 5. Comando de inicio
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "app.main:app", "--bind", "0.0.0.0:8000"]
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código y modelos
COPY . .

# Exponer el puerto
EXPOSE 8000

# Comando para iniciar con Gunicorn (más robusto que Uvicorn solo)
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "app.main:app", "--bind", "0.0.0.0:8000"]