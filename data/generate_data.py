import pandas as pd
import numpy as np
import os

def generar_dataset(num_registros=1000):
    np.random.seed(42) # Para reproducibilidad
    
    data = {
        'id': range(1, num_registros + 1),
        'temp': np.random.uniform(15, 35, num_registros), # Temp en Lima
        'humedad': np.random.uniform(50, 95, num_registros), # Humedad en Lima
    }
    
    df = pd.DataFrame(data)
    
    # Lógica física simulada:
    # A mayor temperatura y humedad, menos días de vida útil.
    # Fórmula: 14 días base - factor de degradación
    df['dias_vida_util'] = 14 - (df['temp'] * 0.3) - (df['humedad'] * 0.05) + np.random.normal(0, 0.5, num_registros)
    
    # Asegurar que no haya vida útil negativa
    df['dias_vida_util'] = df['dias_vida_util'].clip(lower=0)
    
    # Crear ruta de imagen falsa
    df['path_imagen'] = [f'data/images/food_{i}.jpg' for i in range(num_registros)]
    
    # Guardar
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/dataset_alimentos.csv', index=False)
    print(f"Dataset generado con {num_registros} registros en 'data/dataset_alimentos.csv'")

if __name__ == "__main__":
    generar_dataset()