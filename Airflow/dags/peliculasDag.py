from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from datetime import datetime, timedelta
import os

BASE_PATH = "/tmp/movies"
RAW_PATH = f"{BASE_PATH}/movies_raw.json"
FINAL_PATH = f"{BASE_PATH}/movies_final.csv"

default_args = {
    "owner": "martina",
    "start_date": datetime.today() - timedelta(days=1),
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
    "depends_on_past": False,
}

# -----------------------------
# 1️⃣ SIMULAR API DE PELÍCULAS
# -----------------------------
def fetch_movies_from_api(**kwargs):
    import json
    import pandas as pd
    
    os.makedirs(BASE_PATH, exist_ok=True)

    # Simulamos que la API devuelve tu CSV
    df = pd.read_csv("/mnt/data/movies.csv")
    
    movies = df.to_dict(orient="records")

    with open(RAW_PATH, "w") as f:
        json.dump(movies, f)

    return movies


# ---------------------------------
# 2️⃣ LIMPIEZA / TRANSFORMACIÓN
# ---------------------------------
def transform_movies(**kwargs):
    import json
    
    with open(RAW_PATH, "r") as f:
        movies = json.load(f)

    clean_movies = []

    for m in movies:
        clean_movies.append({
            "id": m.get("id"),
            "title": m.get("title"),
            "genre": m.get("genre"),
            "year": int(m.get("year")) if m.get("year") else None,
            "rating": float(m.get("rating")) if m.get("rating") else 0
        })

    kwargs["ti"].xcom_push(key="clean_movies", value=clean_movies)
    return clean_movies


# ---------------------------------
# 3️⃣ SIMULAR API DE ENRIQUECIMIENTO
# ---------------------------------
def enrich_movies(**kwargs):
    import random
    
    ti = kwargs["ti"]
    movies = ti.xcom_pull(task_ids="transform_movies", key="clean_movies")

    enriched = []

    for m in movies:
        m["popularity_score"] = random.randint(1, 100)
        m["estimated_revenue"] = random.randint(1_000_000, 500_000_000)
        enriched.append(m)

    ti.xcom_push(key="enriched_movies", value=enriched)
    return enriched


# ---------------------------------
# 4️⃣ EXPORTAR CSV FINAL
# ---------------------------------
def export_movies(**kwargs):
    import pandas as pd

    ti = kwargs["ti"]
    movies = ti.xcom_pull(task_ids="enrich_movies", key="enriched_movies")

    df = pd.DataFrame(movies)

    df.to_csv(FINAL_PATH, index=False)

    return FINAL_PATH


# =====================
# DEFINICIÓN DEL DAG
# =====================
with DAG(
    dag_id="etl_movies_simulada",
    description="ETL simulando API de películas",
    default_args=default_args,
    schedule=None,
    catchup=False,
    tags=["movies", "etl"],
) as dag:

    start = EmptyOperator(task_id="start")

    fetch_movies = PythonOperator(
        task_id="fetch_movies_from_api",
        python_callable=fetch_movies_from_api,
    )

    transform = PythonOperator(
        task_id="transform_movies",
        python_callable=transform_movies,
    )

    enrich = PythonOperator(
        task_id="enrich_movies",
        python_callable=enrich_movies,
    )

    export = PythonOperator(
        task_id="export_movies",
        python_callable=export_movies,
    )

    start >> fetch_movies >> transform >> enrich >> export