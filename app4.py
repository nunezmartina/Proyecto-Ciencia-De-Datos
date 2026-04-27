import streamlit as st
import pandas as pd
import altair as alt
import base64
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import streamlit.components.v1 as components
from sklearn.neighbors import NearestNeighbors
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
import requests

alt.data_transformers.enable("vegafusion")
alt.renderers.enable("mimetype")

# ========================
# CONFIGURACIÓN GENERAL
# ========================
st.set_page_config(page_title="Explorador de Peliculas", layout="wide")

st.markdown(
    """
    <style>

    .stApp {
        background-color: #141414;
        color: #ffffff;
    }

    section[data-testid="stSidebar"] h1 {
    color: #E50914 !important; /* rojo */
    -webkit-text-stroke: 1px #800020; /* borde */
    font-size: 32px; 
    font-weight: 1000;
    background: none !important;
    -webkit-text-fill-color: #E50914 !important;
}
    h1 {
    background: linear-gradient(90deg, #ff4b5c, #ff6ec7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
    }

    /* 🔥 SUBTITULOS EN BLANCO */
    h2, h3 {
        color: #ffffff;
    }
    p, span, label {
        color: #e5e5e5;
    }

    section[data-testid="stSidebar"] {
        background-color: #000000;
    }

    div[role="radiogroup"] > label {
        padding: 12px 15px;
        margin: 6px 0;
        border-radius: 10px;
        color: #ffffff;
        transition: all 0.2s;
    }

    div[role="radiogroup"] > label:hover {
        background-color: #E50914;
    }

    div[role="radiogroup"] > label:has(input:checked) {
        background-color: #E50914;
        font-weight: bold;
    }

    .stButton > button {
        background-color: #E50914;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 10px 20px;
        font-weight: bold;
    }

    .stButton > button:hover {
        background-color: #b20710;
    }


    </style>
    """,
    unsafe_allow_html=True
)
# ========================
# SIDEBAR DE NAVEGACIÓN
# ========================
st.sidebar.title("Menú")
page = st.sidebar.radio("", ["Introducción al proyecto","Explorador de peliculas", "Exploración libre","Referencias"])


# ==============================
# OPCIÓN EXPLORADOR DE PELICULAS
# ==============================
if page == "Explorador de peliculas":
    st.title("Explorador de Peliculas")

    df = pd.read_csv("movies_final.csv")

    if 'display_name' not in df.columns:
        df['display_name'] = df['title'] + " - " + df['director']

    options_list = sorted(df['display_name'].tolist())

    st.markdown("### Seleccioná una peliculas para explorar sus características:")

    default_track_id = "85f842b8-6817-4721-a85c-8b4dde1e8814"

    if default_track_id in df['imdb_title_id'].values:
        default_display_name = df.loc[df['imdb_title_id'] == default_track_id, 'display_name'].iloc[0]
        default_index = options_list.index(default_display_name) if default_display_name in options_list else 0
    else:

        default_index = 0

    selected_option = st.selectbox("", options_list, index=default_index)
    selected_song = df[df["display_name"] == selected_option].iloc[0]

        # 🔥 FUNCIÓN PARA POSTER
    def get_movie_poster(title):
        url = f"http://www.omdbapi.com/?t={title}&apikey=5af5a66d"
        try:
            response = requests.get(url)
            data = response.json()
            if data["Response"] == "True":
                return data["Poster"]
        except:
            return None
        return None

    # 🔥 OBTENER POSTER
    poster_url = get_movie_poster(selected_song["title"])
    

    st.markdown("---")

    # 🔥 MOSTRAR IMAGEN DE LA PELÍCULA
    col_img, col_info = st.columns([1, 2])

    with col_img:
        if poster_url and poster_url != "N/A":
            st.image(poster_url, width=250)
        else:
            st.warning("No se encontró imagen")

    with col_info:
        st.subheader(selected_song["title"])
        st.write(f" Año: {selected_song['year']}")
        st.write(f" Estreno: {selected_song['date_published']}")
        st.write(f" Director: {selected_song['director']}")
        st.write(f" Actores: {selected_song['actors']}")
        st.write(f" Sinopsis: {selected_song['description']}")
        
    def get_base64_image(image_path):
        with open(image_path, "rb") as img_file:
            encoded = base64.b64encode(img_file.read()).decode()
        return encoded

    genre = selected_song["genre"].lower()
    cluster = selected_song["cluster"]
    backgrounds_generos = {
        "Drama": "images/drama.jpg",
        "Comedy": "images/comedy.jpg",
        "Romance": "images/romance.jpg",
        "Action": "images/action.jpg",
        "Thriller": "images/thriller.jpg",
        "Crime": "images/crime.jpg",
        "Horror": "images/horror.jpg",
        "Adventure": "images/adventure.jpg",
        "Mystery": "images/mystery.jpg",
        "Family": "images/family.jpg",
        "Fantasy": "images/fantasy.jpg",
        "Sci-Fi": "images/scifi.jpg",
        "Animation": "images/animation.jpg",
        "Musical": "images/musical.jpg"
    }
    backgrounds_clusters = {
    0: "images/movido.gif",
    1: "images/aventura.gif",
    2: "images/oscuro.gif"  # o el que quieras
    }

    background_image_generos = backgrounds_generos.get(genre, "images/default.png")
    base64_image_generos = get_base64_image(background_image_generos)

    background_image_clusters = backgrounds_clusters.get(cluster, "images/default.png")
    base64_image_clusters = get_base64_image(background_image_clusters)

    cluster_label = (
    "0 (Emocional-Romantico)" if cluster == 0
    else "1 (Fantastico-Aventurero)" if cluster == 1
    else "2 (Oscuro-Misterioso)"
)

    html_cards = f"""
    <div style="display: flex; justify-content: space-around; margin-top: 20px; margin-bottom: 10px; font-family: 'Source Sans Pro', sans-serif;">
        <div style="position: relative; width: 30%; height: 200px; border-radius: 15px; overflow: hidden; box-shadow: 0px 2px 10px rgba(0,0,0,0.4);">
            <div style="position: absolute; inset: 0; background-image: url('data:image/jpeg;base64,{base64_image_generos}'); background-size: cover; background-position: center; opacity: 0.5;"></div>
            <div style="position: relative; z-index: 1; color: white; text-align: center; font-weight: bold; text-shadow: 1px 1px 4px rgba(0,0,0,0.8); top: 50%; transform: translateY(-50%);">
                <h2>Género</h2>
                <h1>{selected_song['genre']}</h1>
            </div>
        </div>

        <div style="position: relative; width: 30%; height: 200px; border-radius: 15px; overflow: hidden; box-shadow: 0px 2px 10px rgba(0,0,0,0.4);">
            <div style="position: absolute; inset: 0; background-image: url('data:image/gif;base64,{base64_image_clusters}'); background-size: cover; background-position: center; opacity: 0.5;"></div>
            <div style="position: relative; z-index: 1; color: white; text-align: center; font-weight: bold; text-shadow: 1px 1px 4px rgba(0,0,0,0.8); top: 50%; transform: translateY(-50%);">
                <h2>Cluster</h2>
                <h1>{cluster_label}</h1>
            </div>
        </div>

        <div style="position: relative; width: 30%; height: 200px; border-radius: 15px; overflow: hidden; background-color: {'#e57373' if selected_song['anomaly'] == -1 else '#8B4513'}; box-shadow: 0px 2px 10px rgba(0,0,0,0.4); color: white; text-align: center; font-weight: bold; text-shadow: 1px 1px 4px rgba(0,0,0,0.8); display: flex; flex-direction: column; justify-content: center;">
            <h2>Anomalía</h2>
            <h1>{f"Anómala ({selected_song['porcentaje_anomalia']*100:.3f}%)" if selected_song["anomaly"] == -1 else "No anómala"}</h1>
        </div>
    </div>
    """
    features = ['romantic', 'intense', 'sad', 'happy', 'mysterious', 'dark','adventurous', 'fantastical', 'humorous', 'friendly']
    # Mostrar las tarjetas y capturar clics
    components.html(html_cards, height=230)
    
    # Botones para ver las caracteristicas de cada cosa
    col1, col2, col3 = st.columns(3, gap="medium")

    st.markdown("""
    <style>
    div[data-testid="stButton"] {
        display: flex;
        justify-content: center;
    }
    div[data-testid="stButton"] > button {
        min-width: 80%;
        max-width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)
    # Botón GÉNERO 
    with col1:
        inner_left, inner_center, inner_right = st.columns([1, 2, 1])
        with inner_center:
            clicked_genre = st.button(f"Ver más", key="genre_button")

    # Botón CLUSTER 
    with col2:
        inner_left, inner_center, inner_right = st.columns([1, 2, 1])
        with inner_center:
            clicked_cluster = st.button(f"Ver más", key="cluster_button")

    # Botón ANOMALÍA 
    with col3:
        if selected_song["anomaly"] == -1:
            inner_left, inner_center, inner_right = st.columns([1, 2, 1])
            with inner_center:
                clicked_anomaly = st.button("Ver detalles", key="anomaly_button")
            if clicked_anomaly:
                st.session_state.show_anomaly_chart = not st.session_state.get("show_anomaly_chart", False)
        else:
            st.session_state.show_anomaly_chart = False

    # Alternar gráficos 
    if clicked_genre:
        st.session_state.show_genre_chart = not st.session_state.get("show_genre_chart", False)
    if clicked_cluster:
        st.session_state.show_cluster_chart = not st.session_state.get("show_cluster_chart", False)
    
    # Mostrar gráficos 
    if st.session_state.get("show_genre_chart", False):
        st.markdown("---")
        st.subheader(f"Promedio de características del género: {selected_song['genre']}")

        genre_df = df[df["genre"] == selected_song["genre"]]
        genre_avg = genre_df[features].mean().reset_index()
        genre_avg.columns = ["Característica", "Valor promedio"]

        color = "#E66E6E" if cluster == 0 else "#6496E8"

        chart_genre_avg = (
            alt.Chart(genre_avg)
            .mark_bar(color=color, size=25)
            .encode(
                y=alt.Y("Característica:N", sort="-x", title=""),
                x=alt.X("Valor promedio:Q", scale=alt.Scale(domain=[0, 1]), title="Valor promedio"),
                tooltip=["Característica", "Valor promedio"]
            )
            .properties(width=600, height=400,
                        title=f"Promedio de características de las peliculas - {selected_song['genre']}")
        )
        st.altair_chart(chart_genre_avg, use_container_width=True)

    if st.session_state.get("show_cluster_chart", False):
        st.markdown("---")
        st.subheader(f"Promedio de características del cluster {cluster_label}")

        cluster_df = df[df["cluster"] == cluster]
        cluster_avg = cluster_df[features].mean().reset_index()
        cluster_avg.columns = ["Característica", "Valor promedio"]

        color = "#E66E6E" if cluster == 0 else "#6496E8"

        chart_cluster_avg = (
            alt.Chart(cluster_avg)
            .mark_bar(color=color, size=25)
            .encode(
                y=alt.Y("Característica:N", sort="-x", title=""),
                x=alt.X("Valor promedio:Q", scale=alt.Scale(domain=[0, 1]), title="Valor promedio"),
                tooltip=["Característica", "Valor promedio"]
            )
            .properties(width=600, height=400,
                        title=f"Promedio de características de las peliculas - Cluster {cluster_label}")
        )
        st.altair_chart(chart_cluster_avg, use_container_width=True)

    if st.session_state.get("show_anomaly_chart", False):
        st.markdown("---")
        st.subheader("Características con combinaciones inusuales")

        # Calcular correlaciones globales
        corr = df[features].corr()

        # Valores de la canción seleccionada
        song_vals = selected_song[features]

        # Detectar pares conflictivos: correlación negativa + ambos valores altos
        pairs = []
        conflict_features = set()

        for i in range(len(features)):
            for j in range(i+1, len(features)):
                f1, f2 = features[i], features[j]
                corr_val = corr.loc[f1, f2]
                if corr_val < -0.4 and song_vals[f1] > 0.6 and song_vals[f2] > 0.6:
                    pairs.append((f1, f2, corr_val))
                    conflict_features.update([f1, f2])

        if not conflict_features:
            st.info("Esta canción no presenta combinaciones conflictivas destacadas.")
        else:
            conflict_features = list(conflict_features)

        # === Gráfico de barras de características conflictivas ===
        conflict_df = pd.DataFrame({
            "Característica": conflict_features,
            "Valor": [selected_song[f] for f in conflict_features]
        })

        chart_conflicts = (
            alt.Chart(conflict_df)
            .mark_bar(size=40, color="#f5b342")
            .encode(
                x=alt.X("Valor:Q", scale=alt.Scale(domain=[0, 1]), title="Valor de la característica"),
                y=alt.Y("Característica:N", sort="-x", title=""),
                tooltip=["Característica", "Valor"]
            )
            .properties(
                width=500,
                height=350,
                title=alt.TitleParams(
                    text="Características que contribuyen a la anomalía",
                    anchor="middle",
                    fontSize=18,
                    fontWeight=500
                )
            )
        )

        st.altair_chart(chart_conflicts, use_container_width=True)

        st.markdown(
            """
            <div style="text-align:center; font-size:15px; color:gray; margin-top:-10px;">
                Estas características presentan valores altos simultáneamente (generando conflicto),
                aunque suelen estar negativamente correlacionadas.<br>
                Esa combinación poco común hace que la pelicula se considere <strong>inusual</strong>.
            </div>
            """,
            unsafe_allow_html=True
        )




    st.markdown("---")

    # Para caracteristicas de la cancion
    song_features = pd.DataFrame({
        "feature": features,
        "value": [selected_song[f] for f in features]
    })
    color = "#E66E6E" if cluster == 0 else "#6496E8"

    chart_features = (
        alt.Chart(song_features)
        .mark_bar(size=25, color=color)
        .encode(
            x=alt.X("value:Q", title="Valor", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("feature:N", sort="-x", title=""),
            tooltip=["feature", "value"]
        )
        .properties(width=450, height=400)
    )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])
    knn = NearestNeighbors(n_neighbors=6)
    knn.fit(X_scaled)

    idx_selected = df.index[df["display_name"] == selected_option][0]
    distancias, indices = knn.kneighbors([X_scaled[idx_selected]])
    similares_knn = df.iloc[indices[0][1:]][["title","director","genre","cluster"]]
    similares_knn = similares_knn.rename(columns={"title": "Título", "director": "Director", "genre": "Género", "cluster": "Cluster"})

    col_feat, col_sim = st.columns([1, 1])

    with col_feat:
        st.subheader("Características de la pelicula")
        st.altair_chart(chart_features, use_container_width=True)

    with col_sim:
        st.subheader("Peliculas similares")
        st.dataframe(similares_knn, use_container_width=True)

    # ── Segunda tabla: recomendaciones por género, director y año ──
    st.markdown("---")
    st.subheader("Películas recomendadas por género, director y año")

    movie_genre   = selected_song["genre"]
    movie_director = selected_song["director"]
    movie_year    = selected_song["year"]

    # Excluir la película seleccionada
    df_otros = df[df["display_name"] != selected_option].copy()

    # Puntuación: género coincide (+3), director coincide (+2), año cercano (±5 años, +1)
    df_otros["score_contextual"] = 0
    df_otros.loc[df_otros["genre"] == movie_genre,       "score_contextual"] += 3
    df_otros.loc[df_otros["director"] == movie_director, "score_contextual"] += 2
    df_otros.loc[abs(df_otros["year"] - movie_year) <= 5, "score_contextual"] += 1

    # Tomar top 5 con mayor puntuación (desempate por año más cercano)
    df_otros["year_diff"] = abs(df_otros["year"] - movie_year)
    recomendadas_ctx = (
        df_otros[df_otros["score_contextual"] > 0]
        .sort_values(["score_contextual", "year_diff"], ascending=[False, True])
        .head(5)[["title", "director", "genre", "year"]]
        .rename(columns={"title": "Título", "director": "Director",
                         "genre": "Género", "year": "Año"})
        .reset_index(drop=True)
    )

    if recomendadas_ctx.empty:
        st.info("No se encontraron películas con criterios similares de género, director o año.")
    else:
        # Leyenda de criterios usados
        criterios = []
        if (df_otros["genre"] == movie_genre).any():
            criterios.append(f"Género: **{movie_genre}**")
        if (df_otros["director"] == movie_director).any():
            criterios.append(f"Director: **{movie_director}**")
        criterios.append(f"Año cercano a **{movie_year}** (±5 años)")
        st.caption("  ·  ".join(criterios))
        st.dataframe(recomendadas_ctx, use_container_width=True)

    # ── Tercera tabla: TF-IDF + Similitud de Coseno (director, género, año) ──
    st.markdown("---")
    st.subheader("Películas recomendadas por TF-IDF + Similitud de Coseno")
    st.caption("Combina TF-IDF del director y género con el año normalizado para encontrar las películas más similares.")

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import scipy.sparse as sp

    df_tfidf = df.copy().reset_index(drop=True)

    # Normalizar año entre 0 y 1 con peso x0.5
    scaler_year_ctx = MinMaxScaler()
    year_norm = scaler_year_ctx.fit_transform(df_tfidf[["year"]]) * 0.5

    # TF-IDF del director con peso x1.5
    tfidf_dir = TfidfVectorizer()
    dir_matrix = tfidf_dir.fit_transform(df_tfidf["director"].fillna("desconocido")) * 1.5

    # TF-IDF del género con peso x2
    tfidf_genre = TfidfVectorizer()
    genre_matrix = tfidf_genre.fit_transform(df_tfidf["genre"].fillna("desconocido")) * 2.0

    # Año como matriz sparse para poder concatenar
    year_sparse = sp.csr_matrix(year_norm)

    # Combinar todo en una sola matriz sparse
    X_tfidf = sp.hstack([genre_matrix, dir_matrix, year_sparse])

    # Índice de la película seleccionada
    idx_tfidf = df_tfidf.index[df_tfidf["display_name"] == selected_option][0]

    # Calcular similitud de coseno solo contra la película seleccionada
    sim_scores = cosine_similarity(X_tfidf[idx_tfidf], X_tfidf).flatten()

    # Obtener top 5 (excluyendo la misma película)
    sim_scores[idx_tfidf] = -1
    top_indices = sim_scores.argsort()[::-1][:5]

    similares_tfidf = (
        df_tfidf.iloc[top_indices][["title", "director", "genre", "year"]]
        .rename(columns={"title": "Título", "director": "Director",
                         "genre": "Género", "year": "Año"})
        .reset_index(drop=True)
    )

    st.dataframe(similares_tfidf, use_container_width=True)

    st.markdown(
        """
        
        """,
        unsafe_allow_html=True
    )

# ========================
# OPCIÓN REFERENCIAS
# ========================
elif page == "Referencias":
    st.title("Referencias")
    st.markdown("---")

    html_referencias = """
    
<style>
    .container-analisis {
        max-width: 1200px;
        margin: 0 auto;
        padding: 0 20px 40px 20px; 
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    
    .section h2 {
        color: #333333; 
        font-size: 1.5em; 
        margin-bottom: 25px;
        border-bottom: 3px solid #333333; 
        padding-bottom: 10px;
    }

    .section {
        background: #ffffff;
        border-radius: 16px;
        padding: 30px;
        margin-bottom: 30px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.1);
    }
    
    .intro-text {
        font-size: 1.05em;
        line-height: 1.7;
        color: #555555;
        margin-bottom: 20px;
    }

    
    .clusters-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 25px;
        margin-top: 20px;
    }

    .cluster-card {
        border-radius: 12px;
        padding: 25px;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    .cluster-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
    }
    
    .cluster-0 {
        background: linear-gradient(135deg, #FFC4C4 0%, #FFAAAA 100%);
        color: #333333; 
    }

    .cluster-1 {
        background: linear-gradient(135deg, #C4E4FF 0%, #AABEFF 100%);
        color: #333333; 
    }
    
     .cluster-2 {
        background: linear-gradient(135deg, #C8FACC 0%, #6ECC8E 100%);
        color: #333333; 
    }
    .cluster-card h3 {
        font-size: 1.4em;
        margin-bottom: 15px;
        display: flex;
        align-items: center;
        gap: 10px;
        color: #333333;
    }
    
    .cluster-icon {
        font-size: 1.8em;
    }

    .cluster-card p {
        line-height: 1.6;
        font-size: 1em;
    }

    .pca-components {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 25px;
        margin-top: 20px;
    }
    
    .pca-card {
        border-radius: 12px;
        padding: 25px;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }

    .pca-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
    }

    .pca-1 {
        background: linear-gradient(135deg, #e6f7f5 0%, #fff0f5 100%);
    }

    .pca-2 {
        background: linear-gradient(135deg, #fffbe6 0%, #ffe6cc 100%); 
    }

    .characteristics-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 20px;
        overflow: hidden;
        border-radius: 8px;
    }

    .characteristics-table thead {
        background: #f0f0f0; 
        color: #333333;
    }

    .characteristics-table th {
        padding: 12px 18px;
        text-align: left;
        font-weight: 600;
        font-size: 1em;
        border-bottom: 1px solid #cccccc;
    }

    .characteristics-table td {
        padding: 14px 18px;
        border-bottom: 1px solid #e0e0e0;
        line-height: 1.5;
    }

    .characteristics-table tbody tr:hover {
        background-color: #f9f9f9;
    }

    .characteristic-name {
        font-weight: bold; 
        color: #333333; 
        font-size: 1em;
    }
    
    @media (max-width: 768px) {
        .section {
            padding: 15px;
        }

        .section h2 {
            font-size: 1.4em;
        }
    }

    
</style>

<div class="container-analisis">
    <div class="section">
        <h2 id="cluster-title">Análisis de los clusters</h2>
        <p class="intro-text">La técnica de Clustering permitió identificar tres grandes grupos de peliculas según sus características:</p>
        <div class="clusters-grid">
            <div class="cluster-card cluster-0">
                <h3><span class="cluster-icon"></span> Cluster 0 (Emocional-Romantico)</h3>
                <p>Agrupa peliculas con valores más altos en "romantic" y "friendly". Agrupa la mayoría de los géneros porque es el más heterogéneo: Drama,Comedy y Romantic-Comedy. Básicamente acá cae todo el contenido con carga emocional y afectiva que no tiene una característica dominante muy marcada.
            </p>
            </div>
            <div class="cluster-card cluster-1">
                <h3><span class="cluster-icon"></span> Cluster 1 (Fantastico-Aventurero)</h3>
                <p>Reúne peliculas con valores más altos en "fantastical" y "adventurous". En géneros aparecen Action-Adventure, lo que tiene sentido porque tanto las películas de aventura-accion como algunas de terror tienen elementos fantásticos y de mundos imaginarios.
            </p>
            </div>
            <div class="cluster-card cluster-2">
                <h3><span class="cluster-icon"></span> Cluster 2 (Oscuro-Misterioso)</h3>
                <p>Reúne peliculas con valores más altos en "dark" y "mysterious". En géneros predominan Horror-Thriller-Mystery  y Crime, que son exactamente los géneros más oscuros y de suspenso del dataset. Es el cluster con mayor coherencia entre sus características y sus géneros.
            </p>
            </div>
        </div>
        <p class="intro-text" style="margin-top: 25px;">En conjunto, los tres clusters representan tres modos de experiencia en las peliculas predominantes.</p>
    </div>

    <div class="section">
        <h2 id="pca-title">Componentes principales del PCA</h2>
        <p class="intro-text">El análisis de componentes principales permitió reducir las características de las peliculas a las siguientes dos dimensiones:</p>
        <div class="pca-components">
            <div class="pca-card pca-1">
                <h3><span class="pca-icon"></span> Componente 1: Intensidad</h3>
                <p>Está mayormente dominada positivamente por las características `dark` (0.58), `mysterious` (0.55)
                y `adventurous`, mientras que presenta valores negativos en `happy` (-0.32) y `humorous` (-0.25). Esto sugiere que esta componente está asociada a películas con tonos más oscuros, intensos o de suspenso.</p>
            </div>
            <div class="pca-card pca-2">
                <h3><span class="pca-icon"></span> Componente 2: Emocion-Felicidad</h3>
                <p>Está dominada positivamente por `happy` (0.54) y `friendly` (0.58), y negativamente por
            `romantic` (-0.34) y `humorous` (-0.34). Esto podría indicar una dimensión vinculada a contenidos
            más accesibles o familiares, en contraste con otros tonos más emocionales o humorísticos.</p>
            </div>
            
        </div>
        <p class="intro-text" style="margin-top: 25px;">Estas dos dimensiones conforman un mapa que permite visualizar la posición de cada pelicula dentro de los clusters.</p>
    </div>

    <div class="section">
        <h2 id="table-title">Descripción de características</h2>
        <table class="characteristics-table">
            <thead>
                <tr>
                    <th>Característica</th>
                    <th>Descripción</th>
                </tr>
            </thead>
            <tbody>
               <tr>
                    <td class="characteristic-name">happy</td>
                    <td>Indica el nivel de positividad o alegría presente en una película.</td>
                </tr>
                <tr>
                    <td class="characteristic-name">sad</td>
                    <td>Representa el grado de tristeza o carga emocional negativa en una película.</td>
                </tr>
                <tr>
                    <td class="characteristic-name">romantic</td>
                    <td>Mide la presencia de elementos románticos o relaciones amorosas en la historia.</td>
                </tr>
                <tr>
                    <td class="characteristic-name">intense</td>
                    <td>Evalúa el nivel de intensidad emocional o de acción a lo largo de la película.</td>
                </tr>
                <tr>
                    <td class="characteristic-name">mysterious</td>
                    <td>Indica el grado de misterio, intriga o elementos desconocidos en la trama.</td>
                </tr>
                <tr>
                    <td class="characteristic-name">dark</td>
                    <td>Refleja la presencia de temáticas oscuras, tensas o perturbadoras.</td>
                </tr>
                <tr>
                    <td class="characteristic-name">adventurous</td>
                    <td>Mide el nivel de aventura, exploración o viajes dentro de la narrativa.</td>
                </tr>
                <tr>
                    <td class="characteristic-name">fantastical</td>
                    <td>Indica la presencia de elementos fantásticos o sobrenaturales.</td>
                </tr>
                <tr>
                    <td class="characteristic-name">humorous</td>
                    <td>Evalúa el grado de humor o situaciones cómicas en la película.</td>
                </tr>
                <tr>
                    <td class="characteristic-name">friendly</td>
                    <td>Representa qué tan accesible o apta para todo público es la película.</td>
                </tr>
            </tbody>
        </table>
    </div>
    <div class="section">
        <h2 id="genre-title">Descripción sobre géneros</h2>
        <table class="characteristics-table">
            <thead>
                <tr>
                    <th>Género</th>
                    <th>Descripción</th>
                </tr>
            </thead>
            <tbody>
               <tr>
                <td class="characteristic-name">Drama</td>
                <td>Películas centradas en el desarrollo emocional de los personajes y conflictos profundos.</td>
            </tr>
            <tr>
                <td class="characteristic-name">Comedy</td>
                <td>Películas orientadas al humor y entretenimiento, con situaciones divertidas o satíricas.</td>
            </tr>
            <tr>
                <td class="characteristic-name">Romantic-Comedy</td>
                <td>Historias enfocadas en relaciones amorosas y vínculos emocionales entre personajes.</td>
            </tr>
            <tr>
                <td class="characteristic-name">Action-Adeventure</td>
                <td>Películas con alta intensidad, escenas de combate, persecuciones y ritmo dinámico.</td>
            </tr>
            <tr>
                <td class="characteristic-name">Horror-Thriller-Mystery</td>
                <td>Historias de suspenso que generan tensión, misterio y expectativa constante. Incluye generos Horror y Mystery.</td>
            </tr>
            <tr>
                <td class="characteristic-name">Crime</td>
                <td>Películas centradas en delitos, investigaciones policiales y el mundo criminal.</td>
            </tr>
            <tr>
                <td class="characteristic-name">Animation-Family</td>
                <td>Contenido apto para todo público, con valores familiares y temáticas accesibles. Y además agrupa las peliculas de Animation, que serian para niños.</td>
            </tr>
            <tr>
                <td class="characteristic-name">Sci-Fi</td>
                <td>Películas basadas en ciencia ficción, tecnología avanzada o futuros imaginarios.</td>
            </tr>
            <tr>
                <td class="characteristic-name">Musical</td>
                <td>Películas que integran canciones y coreografías como parte central de la narrativa. También en este géenro se encuentra el género Fantasy.</td>
            </tr>
            </tbody>
        </table>
    </div>

    <div class="section">
        <h2 id="anomaly-title">Información sobre peliculas anómalas</h2>
        <div class="pca-components">
            <div class="pca-card pca-1" style="background: #ffe6e6; border: 1px solid #ffaaaa;">
                <h3>¿Qué es una pelicula anómala?</h3>
                <p>Una pelicula es considerada anómala cuando sus características se desvían del patrón general o esperado del resto de los datos.</p>
            </div>
            <div class="pca-card pca-2" style="background: #e6f9ff; border: 1px solid #aad8ff;">
                <h3>Clasificación y porcentaje</h3>
                <p>Utilizamos el algoritmo Isolation Forest para identificar las peliculas anómalas.
                <p>Este porcentaje se encuentra entre 0% (no anómala) y 100% (muy anómala).</p>
            </div>
        </div>
    </div>
</div>

    """

    components.html(html_referencias, height=3700, scrolling=False)

# ===============================
# OPCIÓN INTRODUCCION AL PROYECTO
# ===============================
elif page == "Introducción al proyecto":
    #st.title("MusicApp")
    st.markdown(
        """
        <h1 style="
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            font-weight: 800;
            background: linear-gradient(90deg, #ff6b6b, #5f9eff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: left;
            font-size: 56px;
        ">
        MOVIESENSE
        </h1>
        """,
        unsafe_allow_html=True
    )
    st.subheader("Recomendador de peliculas")


    st.markdown("---")
    html_intro = """
    <style>
        .container-intro {
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 20px 40px 20px;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            color: #333;
        }

        .section {
            background: #ffffff;
            border-radius: 16px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.1);
        }

        .section h2 {
            border-bottom: 3px solid #333333;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }

        .intro-text {
            font-size: 1.05em;
            line-height: 1.7;
            margin-bottom: 20px;
        }
    </style>

    <div class="container-intro">
        <div class="section">
            <h2>Descripción general</h2>
            <p class="intro-text">
            Este proyecto forma parte del <strong>Proyecto Integrador de Ciencia de Datos</strong> y tiene como objetivo 
            construir un sistema de análisis de peliculas que permita explorar, clasificar y recomendar peliculas 
            según sus características.
            </p>

        </div>

        <div class="section">
            <h2>Proceso de desarrollo</h2>
            <p class="intro-text">
            El desarrollo se dividió en distintas etapas:
            </p>
            <ol class="intro-text">
                <li><strong>Extracción y procesamiento de datos:</strong> Airflow automatizó la obtención de peliculas desde APIs externas. Se APIs: IMdb</li>
                <li><strong>Análisis exploratorio:</strong> Realizamos un <em>EDA</em> en Google Colab formulando hipótesis sobre la relación entre las características de las peliculas y sus géneros.</li>
                <li><strong>Aprendizaje supervisado:</strong> Entrenamos distintos modelos para predecir el género de una pelicula, siendo el <em>Random Forest</em> el de mejor desempeño, aunque con limitaciones debido a la cantidad de géneros disponibles.</li>
                <li><strong>Aprendizaje no supervisado:</strong> Aplicamos <em>PCA</em> para reducir dimensiones y descubrimos que lo mejor era utilizar dos componentes. A partir de ellas, se identificaron dos <em>clusters</em>.</li>
                <li><strong>Detección de anomalías:</strong> Usamos el algoritmo <em>Isolation Forest</em> para encontrar alrededor de 300 peliculas con combinaciones inusuales de características.</li>
                <li><strong>Visualización interactiva:</strong> Finalmente, desarrollamos esta aplicación en <em>Streamlit</em> para permitir al usuario explorar peliculas, conocer sus características y descubrir temas similares.</li>
            </ol>
        </div>

        <div class="section">
            <h2>Resultados y aportes</h2>
            <p class="intro-text">
            El sistema permite explorar el espacio de la pelicula de forma visual e interactiva. 
            Los usuarios pueden seleccionar una pelicula, analizar sus atributos, su pertenencia a un cluster 
            y recibir recomendaciones de temas similares.
            </p>
            <p class="intro-text">
            Además, los análisis realizados muestran una coherencia entre las agrupaciones y los géneros.
            Este proyecto sienta las bases para futuras aplicaciones de recomendación de peliculas basadas en contenido.
            </p>
        </div>
    </div>
    """
    components.html(html_intro, height=1500, scrolling=True)

# ========================
# OPCIÓN EXPLORACIÓN LIBRE
# ========================
elif page == "Exploración libre":

    st.title("Análisis y exploración libre de peliculas")
    st.markdown("Explorá los distintos gráficos interactivos creados durante el análisis de datos.")
    st.markdown("---")

    df = pd.read_csv("movies_3000.csv")
    df["display_name"] = df["title"] + " - " + df["director"]
    df_clean = df.dropna(subset=['imdb_title_id', 'cluster'])

    features = ['romantic', 'intense', 'sad', 'happy', 'mysterious', 'dark','adventurous', 'fantastical', 'humorous', 'friendly']

    # Gráfico 1: Clusters y características
    st.subheader("Visualización 1: Clusters y características")

    # Opciones para controles
    genres = ["Todos"] + sorted(df_clean["genre"].dropna().unique().tolist())
    genre_selected = st.selectbox("Filtrar por género:", genres, index=0)

    # Filtrado dinámico
    if genre_selected != "Todos":
        df_filtered = df_clean[df_clean["genre"] == genre_selected]
    else:
        df_filtered = df_clean

    # Selectbox para elegir canción
    song_names = sorted(df_filtered["display_name"].unique().tolist())
    song_selected = st.selectbox("Elegí una pelicula:", song_names)

    selected_song = df_filtered[df_filtered["display_name"] == song_selected].iloc[0]

    # Colores
    cluster_color_scale = alt.Scale(domain=[0, 1, 2], range=['#E66E6E', '#6496E8', '#6ECC8E'])
    color_legend = alt.Legend(title="Tipo de pelicula", labelExpr="datum.value == 0 ? 'Emocional-Romantico' : datum.value == 1 ? 'Fantastico-Aventurero' : 'Oscuro-Misterioso'")

    # Scatter PCA
    scatter = (
        alt.Chart(df_filtered)
        .mark_circle(size=40)
        .encode(
            x=alt.X("pca_1", title="Componente principal 1 (Intensidad)"),
            y=alt.Y("pca_2", title="Componente principal 2 (Emocional-Humor)"),
            color=alt.Color("cluster:N", legend=color_legend, scale=cluster_color_scale),
            tooltip=["title", "director", "genre", "cluster"]
        )
        .properties(width=600, height=500, title=alt.TitleParams(
            text="División de clusters (con PCA)",
            anchor="middle",          
            fontSize=18,
            fontWeight=500
        ))
        .interactive()
    )

    # Canción seleccionada destacada
    highlight = (
        alt.Chart(pd.DataFrame([selected_song]))
        .mark_circle(size=200, color="#f5b342", stroke="black", strokeWidth=2)
        .encode(x="pca_1", y="pca_2", tooltip=["title", "director", "genre"])
    )

    # Gráfico combinado
    chart_pca = scatter + highlight 

    # Características de la canción seleccionada
    song_features = pd.DataFrame({
        "feature": features,
        "value": [selected_song[f] for f in features]
    })

    #colores
    cluster_color_scale_bar = alt.Scale(domain=[0, 1], range=['#E66E6E', '#6496E8'])
    cluster_value_bar = int(selected_song["cluster"])
    cluster_color_bar = "#E66E6E" if cluster_value_bar == 0 else ("#6496E8" if cluster_value_bar == 1 else "#6ECC8E")

    chart_features = (
        alt.Chart(song_features)
        .mark_bar(size=25, color=cluster_color_bar)
        .encode(
            x=alt.X("value:Q", title="Valor", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("feature:N", sort="-x", title=""),
            tooltip=["feature", "value"]
        )
        .properties(width=375, height=375, title=alt.TitleParams(
            text="Características de la pelicula",
            anchor="middle",
            fontSize=18,
            fontWeight=500
        ))
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        st.altair_chart(chart_pca, use_container_width=True)
        st.markdown(
            """
        
            """,
            unsafe_allow_html=True
        )
    with col2:
        st.altair_chart(chart_features, use_container_width=True)
    
    


    st.markdown("---")
    st.subheader("Visualización 2: Géneros y características por cluster")
    
    genre_cluster_counts = (
        df_clean.groupby(['genre', 'cluster'])
        .size()
        .reset_index(name='count')
    )
 
    chart_genre_cluster = (
        alt.Chart(genre_cluster_counts)
        .mark_bar()
        .encode(
            x=alt.X('genre:N', title='Género', sort='-y',
                    axis=alt.Axis(labelAngle=-40, labelOverlap=False)),
            y=alt.Y('count:Q', title='Cantidad de peliculas'),
            color=alt.Color('cluster:N', title='Cluster', scale=cluster_color_scale,
                            legend=alt.Legend(
                                title="Tipo de pelicula",
                                labelExpr="datum.value == 0 ? 'Emocional-Romantico' : datum.value == 1 ? 'Fantastico-Aventurero' : 'Oscuro-Misterioso'"
                            )),
            xOffset='cluster:N',
            tooltip=[
                alt.Tooltip('genre:N', title='Género'),
                alt.Tooltip('cluster:N', title='Cluster'),
                alt.Tooltip('count:Q', title='Cantidad')
            ]
        )
        .properties(
            title=alt.TitleParams(
                text='Distribución de géneros en cada cluster',
                anchor='middle',
                fontSize=18,
                fontWeight=500
            ),
            width=450,
            height=400
        )
    )

    feature_means = (
        df_clean.groupby("cluster")[features].mean().reset_index().melt(id_vars="cluster")
    )

    chart_bar_features = (
        alt.Chart(feature_means)
        .mark_bar()
        .encode(
            y=alt.Y("variable:N", title="Característica de la pelicula", sort='-x'),
            x=alt.X("value:Q", title="Valor promedio"),
            color=alt.Color("cluster:N", title="Cluster", scale=cluster_color_scale,
                            legend=alt.Legend(
                               labelExpr="datum.value == 0 ? 'Emocion' : datum.value == 1 ? 'Fantastico-Aventurero : 'Oscuro-Misterioso'"
                            )),
            xOffset="cluster:N",
            tooltip=["variable:N", "cluster:N", "value:Q"]
        )
        .properties(
            title=alt.TitleParams(
                text="Comparación de características de las peliculas por cluster",
                anchor='middle',
                fontSize=18,
                fontWeight=500
            ),
            width=450,
            height=400
        )
    )

    col1, col2 = st.columns(2)
    with col1:
        st.altair_chart(chart_genre_cluster, use_container_width=True)
    with col2:
        st.altair_chart(chart_bar_features, use_container_width=True)

    
    
    st.markdown("---")
    st.subheader("Visualización 3: Peliculas anómalas")
    st.markdown("Explorá las peliculas más inusuales según sus características detectadas con *Isolation Forest*.")

    # Detección de anomalías
    X = df_clean[features]
    iso_forest = IsolationForest(contamination=0.02, random_state=42)
    predictions = iso_forest.fit_predict(X)
    df_clean['anomaly'] = predictions

    raw_anomaly_scores = iso_forest.score_samples(X)
    inverted_scores = -raw_anomaly_scores.reshape(-1, 1)
    scaler = MinMaxScaler()
    normalized_scores = scaler.fit_transform(inverted_scores)
    df_clean['porcentaje_anomalia'] = normalized_scores

    # Filtro por género
    genres = ["Todos"] + sorted(df_clean["genre"].dropna().unique().tolist())
    genre_selected = st.selectbox("Filtrar por género:", genres, index=0, key="genre_filter_v3")

    if genre_selected == "Todos":
        df_filtered = df_clean.copy()
    else:
        df_filtered = df_clean[df_clean["genre"] == genre_selected]

    # Selección de canción anómala
    df_anomalas = (
        df_filtered[df_filtered["anomaly"] == -1]
        .sort_values(by="porcentaje_anomalia", ascending=False)
        .head(200)
    )

    options_anomalas = df_anomalas["title"] + " - " + df_anomalas["director"]
    selected_song = st.selectbox("Elegí una pelicula anómala:", options_anomalas, key="song_filter_v3")

    selected_row = df_anomalas[df_anomalas["title"] + " - " + df_anomalas["director"] == selected_song].iloc[0]

    # Preparación de gráficos
    cluster_color_scale = alt.Scale(domain=[0, 1, 2], range=["#E66E6E", "#6496E8", "#6ECC8E"])

    color_scale = alt.Scale(domain=[df_anomalas["porcentaje_anomalia"].min(), df_anomalas["porcentaje_anomalia"].max()],
                            range=["#FFD166", "#006400"])
    size_scale = alt.Scale(domain=[df_anomalas["porcentaje_anomalia"].min(), df_anomalas["porcentaje_anomalia"].max()],
                        range=[100, 600])

    # --- Canciones normales ---
    normales = (
        alt.Chart(df_filtered[df_filtered["anomaly"] == 1])
        .mark_circle(size=25)
        .encode(
            x=alt.X("pca_1", title="Componente Principal 1 (Intensidad)"),
            y=alt.Y("pca_2", title="Componente Principal 2 (Emocion-Humor)"),
            color=alt.value("#D3D3D3"),
            opacity=alt.value(0.3),
        )
    )

    # --- Canciones anómalas ---
    anomalas = (
        alt.Chart(df_anomalas)
        .mark_circle()
        .encode(
            x="pca_1",
            y="pca_2",
            fill=alt.Fill("porcentaje_anomalia:Q", scale=color_scale, legend=None),
            size=alt.Size(
                "porcentaje_anomalia:Q",
                scale=size_scale,
                legend=alt.Legend(title="Nivel de Anomalía", format=".0%", titleFontSize=13, labelFontSize=11),
            ),
            tooltip=["title", "director", "genre", alt.Tooltip("porcentaje_anomalia:Q", format=".2%")],
        )
    )

    # --- Canción seleccionada destacada ---
    highlight = (
        alt.Chart(df_filtered[df_filtered["imdb_title_id"] == selected_row["imdb_title_id"]])
        .mark_circle(size=300, color="#f5b342", stroke="black", strokeWidth=2)
        .encode(x="pca_1", y="pca_2", tooltip=["title", "director"])
    )

    # --- Texto con nombre ---
    text_label = (
        alt.Chart(pd.DataFrame([selected_row]))
        .mark_text(
            align="center",
            baseline="bottom",
            dy=-15,
            fontSize=13,
            fontWeight="bold",
            color="black",
        )
        .encode(x="pca_1", y="pca_2", text="title:N")
    )

    # --- Gráfico combinado ---
    scatter_anomalies = (
        alt.layer(normales, anomalas, highlight, text_label)
        .properties(
            title=alt.TitleParams(
                text="Visualización de Anomalías sobre PCA (Top 200 peliculas más anómalas)",
                anchor="middle",
                fontSize=18,
                fontWeight="bold",
            ),
            width=650,
            height=450,
        )
        .interactive()
    )

    # --- Gráfico de características ---
    song_features = pd.DataFrame({
        "feature": features,
        "value": [selected_row[f] for f in features]
    })
    cluster_color_bar = "#E66E6E" if selected_row["cluster"] == 0 else "#6496E8"

    chart_features = (
        alt.Chart(song_features)
        .mark_bar(size=25, color=cluster_color_bar)
        .encode(
            x=alt.X("value:Q", title="Valor", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("feature:N", sort="-x", title=""),
            tooltip=["feature", "value"],
        )
        .properties(width=300, height=400, title="Características de la pelicula seleccionada")
    )


    col1, col2 = st.columns([2, 1])
    with col1:
        st.altair_chart(scatter_anomalies, use_container_width=True)
        st.markdown(
            """
       
            """,
            unsafe_allow_html=True
        )
    with col2:
        st.altair_chart(chart_features, use_container_width=True)

    st.markdown("---")
    st.subheader("Visualización 4: Distribución PCA por género")

    genre_colors = {
        "Drama": "#6D597A",
        "Comedy": "#F4A261",
        "Romantic-Comedy": "#E5989B",
        "Action-Adventure": "#E63946",
        "Horror-Thriller-Mystery": "#264653",
        "Crime": "#2A9D8F",
        "Animation-Family": "#90BE6D",
        "Sci-Fi": "#3A86FF",
        "Musical": "#FF006E"
    }
            
    genres_4 = ["Todos"] + sorted(df_clean["genre"].dropna().unique().tolist())
    genre_selected_4 = st.selectbox("Filtrar por género:", genres_4, index=0, key="genre_filter_v4")

    # --- Filtrado dinámico ---
    if genre_selected_4 != "Todos":
        df_filtered = df_clean[df_clean["genre"] == genre_selected_4]
    else:
        df_filtered = df_clean


    chart_pca_genre = (
        alt.Chart(df_filtered)
        .mark_circle(size=40)
        .encode(
            x=alt.X("pca_1", title="Componente principal 1 (Tranquilidad)"),
            y=alt.Y("pca_2", title="Componente principal 2 (Positividad emocional)"),
            #color=alt.Color("genre_rosamerica:N", legend=alt.Legend(title="Género")),
            color=alt.Color(
                "genre:N",
                legend=alt.Legend(title="Género"),
                scale=alt.Scale(domain=list(genre_colors.keys()), range=list(genre_colors.values()))
            ),
            tooltip=["title", "director", "genre"]
        )
        .properties(
            width=700, 
            height=500, 
            title=(
                f"Proyección PCA - {genre_selected_4 if genre_selected_4 != 'Todos' else 'Todos los géneros'}"
            )
        )
        .interactive()
    )

    st.altair_chart(chart_pca_genre, use_container_width=True)
    st.markdown(
            """
        
            """,
            unsafe_allow_html=True
    )

    st.markdown("---")
    st.subheader("Visualización 5: Gráfico de radar de características por cluster")

    import plotly.graph_objects as go
    features = ["sad", "dark", "happy", "adventurous","mysterious", "fantastical", "humorous", "intense", "friendly", "romantic" ]

    cluster_means = df_clean.groupby("cluster")[features].mean().reset_index()

    fig = go.Figure()

    colors = {0: "#E66E6E", 1: "#6496E8", 2: "#6ECC8E"}

    for i, row in cluster_means.iterrows():
        color = colors[int(row["cluster"])]
        fig.add_trace(go.Scatterpolar(
            r=row[features].values,
            theta=features,
            fill='toself',
            name = (
                         "Emocional-Romantico" if row["cluster"] == 0
                         else "Fantastico-Aventurero" if row["cluster"] == 1
                    else "Oscuro-Misterioso"
            ),
            line=dict(color=color, width=2),
        ))

    fig.update_layout(
        title=dict(
            text="Comparación de características de las peliculas por cluster",
            x=0.5, 
            xanchor="center",
            font=dict(size=18, family="Arial", color="#333", weight=400)
        ),
        polar=dict(radialaxis=dict(visible=True, range=[0,1])),
        showlegend=True,
        height=500,
        width=600
    )
    st.plotly_chart(fig, use_container_width=True)
    

    st.markdown("---")
    st.subheader("Visualización 6: Distribución de géneros y características promedio por género")

    # --- Datos base ---
    genre_counts = df_clean["genre"].value_counts(normalize=False).reset_index()
    genre_counts.columns = ["Género", "Cantidad"]

    # --- Calcular porcentajes ---
    total_songs = genre_counts["Cantidad"].sum()
    genre_counts["Porcentaje"] = genre_counts["Cantidad"] / total_songs

    # Paleta de colores consistente
    genre_colors = {
        "Drama": "#6D597A",
        "Comedy": "#F4A261",
        "Romantic-Comedy": "#E5989B",
        "Action-Adventure": "#E63946",
        "Horror-Thriller-Mystery": "#264653",
        "Crime": "#2A9D8F",
        "Animation-Family": "#90BE6D",
        "Sci-Fi": "#3A86FF",
        "Musical": "#FF006E"
    }
            

    genres_list = ["Todos"] + sorted(df_clean["genre"].dropna().unique().tolist())
    selected_genre = st.selectbox("Elegí un género para analizar sus características:", genres_list, index=0)

    # --- Pie chart base ---
    pie = (
        alt.Chart(genre_counts)
        .mark_arc()
        .encode(
            theta=alt.Theta("Cantidad:Q"),
            order=alt.Order("Porcentaje:Q", sort="descending"),
            color=alt.Color(
                "Género:N",
                scale=alt.Scale(domain=list(genre_colors.keys()), range=list(genre_colors.values())),
                legend=alt.Legend(title="Género")
            ),
            opacity=alt.condition(
                alt.datum.Género == selected_genre,
                alt.value(1.0),
                alt.value(0.85) if selected_genre != "Todos" else alt.value(1.0)
            ),
            stroke=alt.condition(
                alt.datum.Género == selected_genre,
                alt.value("black"),
                alt.value(None)  # sin borde para los demás ni para la leyenda
            ),
            strokeWidth=alt.condition(
                alt.datum.Género == selected_genre,
                alt.value(3),
                alt.value(0)
            ),
            tooltip=["Género", "Cantidad", alt.Tooltip("Porcentaje:Q", format=".1%")]
        )
        .properties(
            width=400,
            height=400,
            title=alt.TitleParams(
                text="Distribución de géneros de peliculas",
                anchor="middle",
                fontSize=18,
                fontWeight=500
            )
        )
    )

    pie_chart = pie 

    # --- Gráfico de barras o mensaje según selección ---
    if selected_genre != "Todos":
        genre_avg = (
            df_clean[df_clean["genre"] == selected_genre][features]
            .mean()
            .reset_index()
        )
        genre_avg.columns = ["Característica", "Valor promedio"]

        chart_genre_avg = (
            alt.Chart(genre_avg)
            .mark_bar(size=30, color=genre_colors.get(selected_genre, "#888"))
            .encode(
                y=alt.Y("Característica:N", sort="-x", title=""),
                x=alt.X("Valor promedio:Q", scale=alt.Scale(domain=[0, 1]), title="Valor promedio"),
                tooltip=["Característica", "Valor promedio"]
            )
            .properties(
                width=450,
                height=400,
                title=alt.TitleParams(
                    text=f"Características promedio - {selected_genre}",
                    anchor="middle",
                    fontSize=18,
                    fontWeight=500
                )
            )
        )
    else:
        msg_html = """
        <div style="display:flex; align-items:center; justify-content:center; height:100%;">
        <div style="text-align:center; color:gray; padding:18px; border-radius:8px;">
            <strong>Seleccioná un género</strong> en el desplegable para ver aquí sus características promedio.
        </div>
        </div>
        """

    # --- Mostrar lado a lado ---
    col1, col2 = st.columns([1, 1])
    with col1:
        st.altair_chart(pie_chart, use_container_width=True)
    with col2:
        if selected_genre != "Todos":
            st.altair_chart(chart_genre_avg, use_container_width=True)
        else:
            st.markdown(msg_html, unsafe_allow_html=True)



    st.markdown("---")
    st.subheader("Visualización 7: Comparación de una característica entre géneros")

    selected_feature = st.selectbox(
        "Elegí una característica para comparar entre géneros:",
        features,
        index=features.index("happy") if "happy" in features else 0
    )

    # Calcular promedios por género
    feature_by_genre = (
        df_clean.groupby("genre")[selected_feature]
        .mean()
        .reset_index()
        .sort_values(by=selected_feature, ascending=False)
    )

    chart_feature_genre = (
        alt.Chart(feature_by_genre)
        .mark_bar(size=25)
        .encode(
            x=alt.X(selected_feature + ":Q", title=f"Promedio de '{selected_feature}'", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("genre:N", title="Género", sort="-x"),
            tooltip=["genre", selected_feature]
        )
        .properties(
            width=600,
            height=400,
            title=alt.TitleParams(
                text=f"Comparación de la característica '{selected_feature}' entre géneros",
                anchor="middle",
                fontSize=18,
                fontWeight=500
            )
        )
    )

    st.altair_chart(chart_feature_genre, use_container_width=True)