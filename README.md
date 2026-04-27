# 🎬 MOVIESENSE — Recomendador de Películas

> Aplicación interactiva de exploración y recomendación de películas desarrollada como Proyecto Integrador de Ciencia de Datos.

🔗 **[Ver aplicación en vivo](https://recomendadordepeliculasmovisense.streamlit.app/)**

---

## 📌 Descripción

**MOVIESENSE** es un sistema de análisis y recomendación de películas que permite explorar, clasificar y recomendar títulos según sus características emocionales, género, director y año. Fue construido utilizando técnicas de aprendizaje automático supervisado y no supervisado, y desplegado como aplicación web con Streamlit.

---

## 🗂️ Secciones de la app

### 🏠 Introducción al proyecto
Descripción general del proyecto, etapas de desarrollo y tecnologías utilizadas.

### 🔍 Explorador de películas
Seleccioná una película y explorá:
- Sus características emocionales (friendly, dark, sad, romantic, etc.) visualizadas en un gráfico de barras
- El cluster al que pertenece (Emocional-Romántico o Fantástico-Aventurero)
- Comparación con el promedio del género y del cluster
- Detección de características inusuales (anomalías)
- **3 sistemas de recomendación:**
  - 🤖 **KNN** sobre características emocionales
  - 📊 **Puntaje por coincidencia** de género, director y año
  - 🧠 **TF-IDF + Similitud de Coseno** sobre género, director y año

### 📊 Exploración libre
7 visualizaciones interactivas del dataset:
1. Clusters y características
2. Géneros y características promedio por cluster
3. Películas anómalas
4. Distribución PCA por género
5. Gráfico de radar por cluster
6. Distribución de géneros y características promedio
7. Comparación de características entre géneros

### 📚 Referencias
Fuentes y bibliografía del proyecto.

---

## 🧠 Técnicas de Machine Learning utilizadas

| Técnica | Uso |
|---|---|
| **KNN** (K-Nearest Neighbors) | Recomendación por características emocionales |
| **TF-IDF + Similitud de Coseno** | Recomendación por género, director y año |
| **PCA** | Reducción de dimensionalidad para clustering |
| **K-Means** | Agrupación de películas en clusters |
| **Isolation Forest** | Detección de películas con características anómalas |
| **Random Forest** | Clasificación supervisada de géneros |

---

## 📁 Estructura del proyecto

```
📦 Proyecto-Ciencia-De-Datos
├── app4.py                  # Aplicación principal Streamlit
├── requirements.txt         # Dependencias
├── movies_final.csv         # Dataset completo (~84k películas)
├── movies_3000.csv          # Dataset reducido para desarrollo
├── images/                  # Imágenes y GIFs por género/cluster
│   ├── drama.jpeg
│   ├── horror.jpeg
│   ├── oscuro.gif
│   └── ...
└── Airflow/                 # Pipeline de extracción de datos
    ├── dags/                # DAGs de Airflow (ETL desde IMDb)
    └── include/             # Scripts auxiliares
```

---

## ⚙️ Instalación local

```bash
# 1. Clonar el repositorio
git clone https://github.com/nunezmartina/Proyecto-Ciencia-De-Datos.git
cd Proyecto-Ciencia-De-Datos

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Ejecutar la app
streamlit run app4.py
```

---

## 📦 Dependencias

```
streamlit
pandas
numpy
scikit-learn
altair
vegafusion
requests
scipy
plotly
```

---

## 🔄 Pipeline de datos (Airflow)

La extracción de datos fue automatizada con **Apache Airflow**, consumiendo la API de **IMDb** para obtener información de películas: título, director, actores, género, año, sinopsis y características emocionales.

---

##
Proyecto Integrador — Ingeniería en Ciencia de Datos  

---
