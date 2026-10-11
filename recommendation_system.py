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

# Definición de estilos y colores ANSI para la consola
class UI:
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    RESET = '\033[0m'

    ICONS = {
        'Smartphones': '📱 Smartphones',
        'Laptops': '💻 Laptops',
        'Accesorios': '🔌 Accesorios',
        'Smartwatches': '⌚ Smartwatches',
        'Audio': '🎧 Audio'
    }

def print_banner():
    print(f"\n{UI.CYAN}{UI.BOLD}╔══════════════════════════════════════════════════════════════════════════════════╗{UI.RESET}")
    print(f"{UI.CYAN}{UI.BOLD}║         🤖  SISTEMA DE RECOMENDACIÓN INTELIGENTE CON MACHINE LEARNING            ║{UI.RESET}")
    print(f"{UI.CYAN}{UI.BOLD}║               INACAP  •  ETVI02 - Tendencias Emergentes en IA                    ║{UI.RESET}")
    print(f"{UI.CYAN}{UI.BOLD}╚══════════════════════════════════════════════════════════════════════════════════╝{UI.RESET}\n")

def load_or_create_dataset():
    """Carga los datos desde products.csv o genera un dataset sintético si no existe."""
    dataset_path = os.path.join(os.path.dirname(__file__), 'products.csv')
    if os.path.exists(dataset_path):
        data = pd.read_csv(dataset_path)
        print(f" {UI.GREEN}✓{UI.RESET} {UI.BOLD}Dataset cargado:{UI.RESET} {dataset_path} ({len(data)} registros)")
        return data
    
    print(f" {UI.YELLOW}!{UI.RESET} Dataset no encontrado localmente. Generando datos de muestra...")
    synthetic_data = {
        'price_normalized': [0.68, 0.95, 0.15, 0.48, 0.32],
        'rating': [4.7, 4.9, 4.1, 4.5, 4.4],
        'popularity': [0.92, 0.82, 0.40, 0.72, 0.60],
        'category': ['Smartphones', 'Laptops', 'Accesorios', 'Smartwatches', 'Audio']
    }
    return pd.DataFrame(synthetic_data)

def main():
    print_banner()

    # 1. Cargar los datos
    data = load_or_create_dataset()

    # 2. Preprocesamiento de datos
    features = data[['price_normalized', 'rating', 'popularity']]
    labels = data['category']

    # 3. Dividir datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.25, random_state=42
    )

    # 4. Crear y entrenar el modelo (K-Nearest Neighbors)
    k_neighbors = 3
    model = KNeighborsClassifier(n_neighbors=k_neighbors)
    model.fit(X_train, y_train)
    print(f" {UI.GREEN}✓{UI.RESET} {UI.BOLD}Algoritmo entrenado:{UI.RESET} KNeighborsClassifier (k={k_neighbors} vecinos)")

    # 5. Realizar predicciones y evaluar métricas
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    progress_bar = "█" * int(accuracy * 20)
    print(f" {UI.GREEN}✓{UI.RESET} {UI.BOLD}Precisión del Modelo (Accuracy):{UI.RESET} {UI.GREEN}{progress_bar} {accuracy * 100:.1f}%{UI.RESET}")

    # 6. Función de recomendación
    def recommend(product_features):
        input_df = pd.DataFrame([product_features], columns=['price_normalized', 'rating', 'popularity'])
        prediction = model.predict(input_df)
        return prediction[0]

    # 7. Demostración de recomendaciones en tabla visual
    test_cases = [
        {"name": "Dispositivo móvil gama alta", "features": [0.67, 4.6, 0.91]},
        {"name": "Computador portátil trabajo", "features": [0.93, 4.8, 0.82]},
        {"name": "Cable USB / Adaptador", "features": [0.15, 4.0, 0.42]},
        {"name": "Smartwatch deportivo", "features": [0.49, 4.6, 0.76]},
        {"name": "Auriculares inalámbricos", "features": [0.31, 4.4, 0.61]}
    ]

    print(f"\n{UI.YELLOW}{UI.BOLD}📊 DEMOSTRACIÓN DE RECOMENDACIONES EN TIEMPO REAL:{UI.RESET}")
    print(f"{UI.DIM}┌───────────────────────────────┬─────────────────────────────┬───────────────────────────┐{UI.RESET}")
    print(f"{UI.BOLD}│ Producto a Evaluar            │ Características [P / R / Pop]│ Categoría Recomendada    │{UI.RESET}")
    print(f"{UI.DIM}├───────────────────────────────┼─────────────────────────────┼───────────────────────────┤{UI.RESET}")

    for case in test_cases:
        category = recommend(case["features"])
        category_formatted = UI.ICONS.get(category, category)
        p, r, pop = case["features"]
        feat_str = f"P:{p:.2f} | R:{r:.1f}★ | Pop:{pop:.2f}"
        print(f"│ {case['name']:<29} │ {feat_str:<27} │ {UI.BOLD}{UI.CYAN}{category_formatted:<25}{UI.RESET} │")

    print(f"{UI.DIM}└───────────────────────────────┴─────────────────────────────┴───────────────────────────┘{UI.RESET}")
    print(f"\n{UI.GREEN}✨ Sistema inicializado con éxito y listo para predicciones.{UI.RESET}\n")

if __name__ == '__main__':
    main()
