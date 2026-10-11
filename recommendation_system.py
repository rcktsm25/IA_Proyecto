"""
Sistema de Recomendación y Clasificación de Productos
Actividad Formativa: GitHub Copilot (ETVI02_U3_S8_AF)
"""

# Importar las librerías necesarias
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

def load_or_create_dataset():
    """Carga los datos desde products.csv o genera un dataset sintético si no existe."""
    dataset_path = os.path.join(os.path.dirname(__file__), 'products.csv')
    if os.path.exists(dataset_path):
        print(f"Cargando dataset desde: {dataset_path}")
        return pd.read_csv(dataset_path)
    
    print("Dataset no encontrado localmente. Generando datos sintéticos de muestra...")
    synthetic_data = {
        'price_normalized': [0.85, 0.78, 0.90, 0.65, 0.95, 0.88, 0.70, 0.82, 0.25, 0.15, 0.30, 0.20, 0.50, 0.55, 0.45, 0.48, 0.40, 0.35, 0.60, 0.38],
        'rating': [4.7, 4.5, 4.8, 4.2, 4.9, 4.6, 4.3, 4.4, 4.1, 3.9, 4.3, 4.0, 4.6, 4.7, 4.3, 4.4, 4.5, 4.2, 4.8, 4.3],
        'popularity': [0.92, 0.88, 0.95, 0.75, 0.98, 0.89, 0.80, 0.84, 0.60, 0.50, 0.68, 0.55, 0.85, 0.87, 0.78, 0.82, 0.91, 0.77, 0.93, 0.80],
        'category': ['Smartphones', 'Smartphones', 'Smartphones', 'Smartphones', 'Laptops', 'Laptops', 'Laptops', 'Laptops',
                     'Accesorios', 'Accesorios', 'Accesorios', 'Accesorios', 'Smartwatches', 'Smartwatches', 'Smartwatches', 'Smartwatches',
                     'Audio', 'Audio', 'Audio', 'Audio']
    }
    return pd.DataFrame(synthetic_data)

def main():
    print("=== Inicializando Sistema de Recomendación ===")
    
    # Cargar los datos
    data = load_or_create_dataset()
    print(f"Total de registros cargados: {len(data)}")
    
    # Preprocesamiento de datos
    # Seleccionar las características y la etiqueta
    features = data[['price_normalized', 'rating', 'popularity']]
    labels = data['category']

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.25, random_state=42
    )

    # Crear y entrenar el modelo con KNeighborsClassifier (K-Nearest Neighbors)
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)
    print("Modelo entrenado exitosamente con algoritmo KNeighborsClassifier.")

    # Realizar predicciones sobre el conjunto de prueba
    y_pred = model.predict(X_test)

    # Evaluar el modelo
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Precisión del modelo (Accuracy): {accuracy * 100:.2f}%\n")

    # Función de recomendación basada en características del producto
    def recommend(product_features):
        """
        Predice y recomienda la categoría de producto basada en sus características.
        product_features: [price_normalized, rating, popularity]
        """
        prediction = model.predict([product_features])
        return prediction[0]

    # Casos de prueba / Ejemplo de recomendación
    test_cases = [
        {"name": "Dispositivo móvil gama alta", "features": [0.82, 4.6, 0.91]},
        {"name": "Computador portátil para trabajo", "features": [0.92, 4.8, 0.95]},
        {"name": "Cable USB / Adaptador", "features": [0.18, 4.0, 0.52]},
        {"name": "Auriculares inalámbricos", "features": [0.42, 4.4, 0.88]}
    ]

    print("--- Demostración de Recomendaciones ---")
    for case in test_cases:
        category = recommend(case["features"])
        print(f"Producto: {case['name']}")
        print(f"  Características [Precio norm, Rating, Popularidad]: {case['features']}")
        print(f"  Categoría recomendada: {category}\n")

if __name__ == '__main__':
    main()
