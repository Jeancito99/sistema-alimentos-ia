# Usa una imagen base de Python
FROM python:3.10-slim

# Directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema necesarias para Keras/TensorFlow
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependencias de Python
COPY requirements.txt .
COPY app/ ./app/

# 2. Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt


# 4. Copiar el resto del código si es necesario
COPY . .
# Variable importante para Render
ENV PYTHONUNBUFFERED=1

# Comando para iniciar con Gunicorn (más robusto que Uvicorn solo)
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port $PORT"]