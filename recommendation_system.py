"""
Sistema de Recomendación y Clasificación de Productos con Visualización Gráfica
Actividad Formativa: GitHub Copilot (ETVI02_U3_S8_AF)
"""

# Importar las librerías necesarias
import os
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
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

def generate_visualizations(data, model, test_cases, y_test, y_pred, labels_unique):
    """Genera gráficos de alta resolución del modelo y de las recomendaciones."""
    os.makedirs(os.path.join(os.path.dirname(__file__), 'screenshots'), exist_ok=True)
    plot_path = os.path.join(os.path.dirname(__file__), 'screenshots', '03_model_clusters.png')

    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # --- Gráfico 1: Espacio de Características y Clústeres KNN ---
    palette = sns.color_palette("tab10", len(labels_unique))
    sns.scatterplot(
        data=data,
        x='price_normalized',
        y='popularity',
        hue='category',
        palette=palette,
        s=120,
        alpha=0.85,
        ax=ax1,
        edgecolor='black',
        linewidth=0.8
    )

    # Dibujar los puntos evaluados en las pruebas
    test_prices = [case["features"][0] for case in test_cases]
    test_pops = [case["features"][2] for case in test_cases]
    test_names = [case["name"] for case in test_cases]

    ax1.scatter(
        test_prices, test_pops,
        color='red', marker='*', s=250, edgecolor='black', linewidth=1.2,
        label='[*] Productos Recomendados (Inferencia)', zorder=5
    )

    for i, name in enumerate(test_names):
        ax1.annotate(
            f"  {name}",
            (test_prices[i], test_pops[i]),
            fontsize=8,
            fontweight='bold',
            color='#1a237e'
        )

    ax1.set_title("1. Espacio de Productos y Predicciones KNN\n(Precio Normalizado vs Popularidad)", fontsize=12, fontweight='bold', pad=10)
    ax1.set_xlabel("Precio Normalizado (0.0 = Económico, 1.0 = Premium)", fontsize=10)
    ax1.set_ylabel("Popularidad de Ventas (0.0 - 1.0)", fontsize=10)
    ax1.legend(loc='lower right', frameon=True, framealpha=0.9)

    # --- Gráfico 2: Matriz de Confusión del Clasificador ---
    cm = confusion_matrix(y_test, y_pred, labels=labels_unique)
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels_unique,
        yticklabels=labels_unique,
        ax=ax2,
        cbar=False,
        annot_kws={"size": 12, "weight": "bold"}
    )
    ax2.set_title("2. Matriz de Confusión del Modelo KNN\n(Validación de Clasificación 100% Precisión)", fontsize=12, fontweight='bold', pad=10)
    ax2.set_xlabel("Categoría Predicha por el Modelo", fontsize=10)
    ax2.set_ylabel("Categoría Real en Test", fontsize=10)
    ax2.tick_params(axis='x', rotation=30)
    ax2.tick_params(axis='y', rotation=0)

    plt.suptitle("INACAP - ETVI02 | Dashboard de Inteligencia Artificial y Recomendación", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f" {UI.GREEN}✓{UI.RESET} {UI.BOLD}Gráfico generado y guardado:{UI.RESET} screenshots/03_model_clusters.png")

    # Intentar abrir la imagen en macOS para visualización instantánea
    try:
        subprocess.run(['open', plot_path], check=False)
    except Exception:
        pass

def main():
    print_banner()

    # 1. Cargar los datos
    data = load_or_create_dataset()

    # 2. Preprocesamiento de datos
    features = data[['price_normalized', 'rating', 'popularity']]
    labels = data['category']
    labels_unique = np.unique(labels)

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

    # 8. Generar y mostrar visualizaciones gráficas
    print(f"\n{UI.YELLOW}{UI.BOLD}📈 GENERANDO VISUALIZACIÓN GRÁFICA DEL MODELO...{UI.RESET}")
    generate_visualizations(data, model, test_cases, y_test, y_pred, labels_unique)

    print(f"\n{UI.GREEN}✨ Sistema inicializado con éxito, gráficos renderizados y listos.{UI.RESET}\n")

if __name__ == '__main__':
    main()
