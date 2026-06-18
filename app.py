import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
import base64

# 1. Configuración de la Página
st.set_page_config(page_title="Real Madrid UCL Evolution", layout="wide")

# HEADER: TÍTULO E IMÁGENES
# ==============================================================================
# Título Principal Centrado
st.markdown("<h1 style='text-align: center; margin-top: 10px; margin-bottom: 25px;'>⚪ Real Madrid: UCL Performance Dashboard ⚪</h1>", unsafe_allow_html=True)

# ==============================================================================
# IMAGEN DE BIENVENIDA Y GALERÍA SECUNDARIA (ORDEN CORREGIDO)
# ==============================================================================
def obtener_imagen_base64(ruta):
    with open(ruta, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

try:
    # 1. Imagen Principal: Florentino ocupando todo el ancho
    img_florentino = obtener_imagen_base64("florentino.webp")
    st.markdown(
        f"""
        <div style="width: 100%; overflow: hidden; margin-bottom: 10px;">
            <img src="data:image/webp;base64,{img_florentino}" style="width: 100%; height: auto; object-fit: cover; display: block; border-radius: 4px;">
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # 2. Rutas de acceso absolutas
    ruta_museo = "florentino-perez-museo-champions-league_117_1200x675.jpeg"
    ruta_madrid = "Real-Madrid-1024x684.png"
    
    # Cargar datos binarios
    img_museo_data = obtener_imagen_base64(ruta_museo)
    img_madrid_data = obtener_imagen_base64(ruta_madrid)
    
    # Creamos las dos columnas para el ancho completo de la página
    col_izq_galeria, col_der_galeria = st.columns(2)
    
    # Altura fija compartida para garantizar simetría visual
    altura_galeria = "350px"
    
    # PRIMERO: Florentino Museo Champions League (Izquierda)
    with col_izq_galeria:
        st.markdown(
            f"""
            <div style="width: 100%; overflow: hidden; margin-bottom: 20px;">
                <img src="data:image/jpeg;base64,{img_museo_data}" 
                     style="width: 100%; height: {altura_galeria}; object-fit: cover; display: block; border-radius: 4px;">
            </div>
            """, 
            unsafe_allow_html=True
        )
        
    # LUEGO: Real-Madrid-1024x684.png (Derecha con auto-recorte inferior)
    with col_der_galeria:
        st.markdown(
            f"""
            <div style="width: 100%; overflow: hidden; margin-bottom: 20px;">
                <img src="data:image/png;base64,{img_madrid_data}" 
                     style="width: 100%; height: {altura_galeria}; object-fit: cover; object-position: top; display: block; border-radius: 4px;">
            </div>
            """, 
            unsafe_allow_html=True
        )

except FileNotFoundError as e:
    st.warning(f"Error de ruta: No se pudo cargar uno de los archivos. Detalles: {e}")
# ==============================================================================
st.markdown("<br><hr>", unsafe_allow_html=True) # Separador estético
st.markdown("A unified analysis evaluating overall campaign trajectories, historical stages reached, and structural player distributions under Florentino Perez presidency.")


# 2. Comprehensive Multi-Era Data Processing
@st.cache_data
def load_historical_database():
    # Historical Performance Tracking (2000 - 2026)
    # Stage mapping: 1 = Round of 16, 2 = Quarter-Finals, 3 = Semi-Finals, 5 = Winner
    timeline_data = {
        "Temporada": list(range(2000, 2027)),
        "Goles_Plantilla": [
            30, 35, 35, 26, 16, 14, 14,             # 2000 - 2006
            None, None, None,                       # 2007 - 2009 (Presidential Gap)
            24, 25, 35, 26, 41, 24, 28, 34, 33,     # 2010 - 2018
            16, 16, 19, 29, 26, 26, 29, 32          # 2019 - 2026
        ],
        "Fase_Numerica": [
            5, 3, 5, 3, 2, 1, 1,                    # 2000 - 2006
            None, None, None,                       # 2007 - 2009 (Presidential Gap)
            1, 3, 3, 3, 5, 3, 5, 5, 5,              # 2010 - 2018
            1, 1, 3, 5, 3, 5, 2, 2                  # 2019 - 2026
        ],
        "Resultado": [
            "Winner", "Semi-Final", "Winner", "Semi-Final", "Quarter-Final", "Round of 16", "Round of 16",
            "Brecha", "Brecha", "Brecha",
            "Round of 16", "Semi-Final", "Semi-Final", "Semi-Final", "Winner", "Semi-Final", "Winner", "Winner", "Winner",
            "Round of 16", "Round of 16", "Semi-Final", "Winner", "Semi-Final", "Winner", "Quarter-Final", "Quarter-Final"
        ]
    }
    df_time = pd.DataFrame(timeline_data)
    
    # Combined Player Matrix (Scraped from all eras)
    player_data = {
        "Jugador": [
            "Cristiano Ronaldo (16/17)", "Karim Benzema (16/17)", "Álvaro Morata (16/17)", "Marco Asensio (16/17)",
            "Gareth Bale (16/17)", "Casemiro (16/17)", "Sergio Ramos (16/17)", "Toni Kroos (16/17)",
            "Luka Modrić (16/17)", "Dani Carvajal (16/17)", "Vinícius Júnior (23/24)", "Jude Bellingham (23/24)",
            "Rodrygo (23/24)", "Joselu (23/24)", "Kylian Mbappé (25/26)", "Federico Valverde (25/26)",
            "Arda Güler (25/26)", "Aurélien Tchouaméni (25/26)", "Brahim Díaz (25/26)"
        ],
        "Goles": [12, 5, 3, 3, 2, 2, 1, 1, 0, 0, 6, 4, 5, 5, 15, 3, 2, 1, 1],
        "Asistencias": [6, 2, 1, 1, 2, 2, 3, 2, 1, 5, 4, 5, 2, 0, 1, 4, 4, 1, 2],
        "Minutos": [1200, 954, 165, 253, 509, 827, 1001, 1027, 983, 976, 898, 989, 1012, 282, 912, 1161, 1019, 1123, 416]
    }
    df_players = pd.DataFrame(player_data)
    df_players["G+A"] = df_players["Goles"] + df_players["Asistencias"]
    
    return df_time, df_players

df_time, df_players = load_historical_database()

# 3. K-Means Strategic Clustering
X = df_players[["Goles", "Asistencias"]]
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X)
df_players["Cluster_ID"] = kmeans.labels_
cluster_labels = {0: "Support Specialists", 1: "Elite Attackers", 2: "Playmakers / Creators"}
df_players["Tactical Profile"] = df_players["Cluster_ID"].map(cluster_labels)

# 4. Dashboard Sections
# SECTION 1: CAMPAIGN EVOLUTION & PROGRESSION
st.header("📈 Historical Evolution under Florentino Perez mandate (2000-2026)")
col1, col2 = st.columns(2)

with col1:
    # New Requested Graph: Stage Reached Evolution
    fig_stage = px.line(df_time, x="Temporada", y="Fase_Numerica", markers=True,
                        title="Maximum Stage Reached per Tournament",
                        color_discrete_sequence=["#0284C7"],
                        hover_data={"Resultado": True, "Fase_Numerica": False})
    fig_stage.update_traces(connectgaps=False, line=dict(width=3.5), marker=dict(size=9))
    fig_stage.update_layout(
        yaxis=dict(
            tickvals=[1, 2, 3, 5],
            ticktext=["Round of 16", "Quarter-Finals", "Semi-Finals", "🏆 Winner"]
        ),
        xaxis=dict(tickmode='linear', dtick=2),
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis_title="Stage Achieved"
    )
    st.plotly_chart(fig_stage, use_container_width=True)

with col2:
    # Campaign Goals Timeline
    fig_goals = px.line(df_time, x="Temporada", y="Goles_Plantilla", markers=True,
                        title="Total Goals Scored per Season",
                        color_discrete_sequence=["#1E3A8A"])
    fig_goals.update_traces(connectgaps=False, line=dict(width=3), marker=dict(size=8))
    fig_goals.update_layout(xaxis=dict(tickmode='linear', dtick=2), plot_bgcolor="rgba(0,0,0,0)", yaxis_title="Goals")
    st.plotly_chart(fig_goals, use_container_width=True)

st.info("💡 **Presidential Transition Note:** The continuous blank timeline gap from **2007 to 2009** reflects the non-Florentino Pérez management period, keeping your comparative analysis strictly isolated.")

st.markdown("---")

# SECTION 2: ATTACKING CONTRIBUTION & CLUSTERS
st.header("🎯 Player's Metrics Analytics")
col3, col4 = st.columns(2)

with col3:
    fig_bar = px.bar(df_players.sort_values(by="G+A", ascending=True), 
                     x="G+A", y="Jugador", orientation='h', color="Goles",
                     title="Total Direct Contributions (Goals + Assists)",
                     color_continuous_scale=px.colors.sequential.Cividis)
    st.plotly_chart(fig_bar, use_container_width=True)

with col4:
    fig_scatter = px.scatter(df_players, x="Goles", y="Asistencias", color="Tactical Profile",
                             size="Minutos", hover_name="Jugador",
                             title="K-Means Tactical Grouping",
                             color_discrete_sequence=["#64748B", "#D97706", "#2563EB"])
    st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown("---")

# SECTION 3: TABULAR DATA SUMMARY
st.header("📋 Compiled Dataset Explorer")
st.dataframe(df_players[["Jugador", "Goles", "Asistencias", "G+A", "Minutos", "Tactical Profile"]].sort_values(by="G+A", ascending=False), use_container_width=True)


# Ruta absoluta al nuevo archivo AVIF en tu entorno
ruta_bernabeu_avif = "/workspaces/RealMadrid-FlorentinoPresidency/bernabeu.avif"

try:
    with open(ruta_bernabeu_avif, "rb") as file:
        img_base64 = base64.b64encode(file.read()).decode()

    # Renderizado en formato AVIF, tamaño original y sin márgenes adicionales
    st.markdown(
        f"""
        <div style="width: 100%; margin: 0; padding: 0;">
            <img src="data:image/avif;base64,{img_base64}" 
                 style="width: 100%; height: auto; display: block; border-radius: 6px; margin: 0; padding: 0;">
        </div>
        """, 
        unsafe_allow_html=True
    )
except FileNotFoundError:
    st.error(f"No se encontró el archivo 'bernabeu.avif' en la ruta: {ruta_bernabeu_avif}")


# Mostrar el Escudo Pequeño y Centrado
col_izq_escudo, col_centro_escudo, col_der_escudo = st.columns([3, 1, 3])
with col_centro_escudo:
    st.image("escudo madrid.png", use_container_width=True)

# ==============================================================================
# CRÉDITOS FINALES
# ==============================================================================
st.markdown(
    """
    <p style="text-align: center; font-size: 0.85rem; color: #888888; margin-top: 40px; margin-bottom: 10px;">
        Proyecto de Valentín Gerold en colaboración con Gemini IA
    </p>
    """, 
    unsafe_allow_html=True
)
