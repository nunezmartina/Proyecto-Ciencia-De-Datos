from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import os
import requests
import zipfile
import io

# ===================== Configuración =====================
DATA_DIR = "/tmp/movies"
RAW_PATH = f"{DATA_DIR}/raw"
STAGING_PATH = f"{DATA_DIR}/staging"
OUTPUT_PATH = "/usr/local/airflow/include/output"
MOVIELENS_URL = "http://files.grouplens.org/datasets/movielens/ml-latest-small.zip"

default_args = {
    "owner": "grupo",
    "start_date": datetime(2025, 9, 1),
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
    "depends_on_past": False,
}

# ===================== Funciones =====================
def extract_data():
    os.makedirs(RAW_PATH, exist_ok=True)
    resp = requests.get(MOVIELENS_URL, timeout=60)
    if resp.status_code == 200:
        with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
            z.extractall(RAW_PATH)
    else:
        raise Exception(f"Error al descargar datos: {resp.status_code}")

def load_raw():
    os.makedirs(STAGING_PATH, exist_ok=True)
    source_file = os.path.join(RAW_PATH, "ml-latest-small", "movies.csv")
    dest_file = os.path.join(STAGING_PATH, "movies_raw.csv")
    pd.read_csv(source_file).to_csv(dest_file, index=False)

def transform_data():
    staging_file = os.path.join(STAGING_PATH, "movies_raw.csv")
    df = pd.read_csv(staging_file)

    # Año y título
    df["year"] = pd.to_numeric(df["title"].str.extract(r"\((\d{4})\)")[0], errors="coerce").astype("Int64")
    df["title_clean"] = (
        df["title"].str.replace(r"\(\d{4}\)", "", regex=True)
                   .str.replace(r"\s+", " ", regex=True)
                   .str.strip()
    )

    # Géneros
    def clean_genres(g):
        if pd.isna(g): return ""
        g = str(g).strip()
        if g.lower() == "(no genres listed)": return ""
        parts = [p.strip() for p in g.split("|") if p.strip()]
        seen = set(); uniq = [p for p in parts if not (p in seen or seen.add(p))]
        return "|".join(uniq)

    df["genres"] = df["genres"].map(clean_genres)
    df["genres_list"] = df["genres"].apply(lambda s: [p for p in s.split("|") if p] if s else [])

    # Selección + orden
    df_clean = (df[["movieId","title_clean","year","genres","genres_list"]]
                .sort_values(["title_clean","year","movieId"])
                .drop_duplicates(subset=["movieId"])
                .reset_index(drop=True))

    out_tmp = os.path.join(STAGING_PATH, "movies_clean.csv")
    df_clean.to_csv(out_tmp, index=False)

def transform_ratings():
    source_file = os.path.join(RAW_PATH, "ml-latest-small", "ratings.csv")
    df = pd.read_csv(source_file)

    # Estadísticas por película
    stats = df.groupby("movieId").agg(
        n_ratings=("rating", "count"),
        mean_rating=("rating", "mean")
    ).reset_index()

    # Popularidad bayesiana
    global_mean = df["rating"].mean()
    m = 50  # parámetro de suavizado
    stats["popularity_score"] = (
        (stats["n_ratings"] / (stats["n_ratings"] + m)) * stats["mean_rating"]
        + (m / (stats["n_ratings"] + m)) * global_mean
    )

    out_tmp = os.path.join(STAGING_PATH, "ratings_stats.csv")
    stats.to_csv(out_tmp, index=False)

def join_datasets():
    movies_file = os.path.join(STAGING_PATH, "movies_clean.csv")
    ratings_file = os.path.join(STAGING_PATH, "ratings_stats.csv")

    movies = pd.read_csv(movies_file)
    ratings = pd.read_csv(ratings_file)

    merged = movies.merge(ratings, on="movieId", how="left")
    merged[["n_ratings","mean_rating","popularity_score"]] = merged[
        ["n_ratings","mean_rating","popularity_score"]
    ].fillna(0)

    out_file = os.path.join(STAGING_PATH, "movies_with_ratings.csv")
    merged.to_csv(out_file, index=False)

def export_data():
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    source_file = os.path.join(STAGING_PATH, "movies_with_ratings.csv")
    dest_file = os.path.join(OUTPUT_PATH, "movies_with_ratings.csv")
    pd.read_csv(source_file).to_csv(dest_file, index=False)

# ===================== DAG =====================
with DAG(
    dag_id="movies_pipeline_v2",
    default_args=default_args,
    description="Pipeline de películas + ratings",
    schedule=None,
    catchup=False,
) as dag:

    t1 = PythonOperator(task_id="extract_data", python_callable=extract_data)
    t2 = PythonOperator(task_id="load_raw", python_callable=load_raw)
    t3 = PythonOperator(task_id="transform_data", python_callable=transform_data)
    t4 = PythonOperator(task_id="transform_ratings", python_callable=transform_ratings)
    t5 = PythonOperator(task_id="join_datasets", python_callable=join_datasets)
    t6 = PythonOperator(task_id="export_data", python_callable=export_data)

    t1 >> t2
    t2 >> t3
    t1 >> t4
    [t3, t4] >> t5 >> t6
