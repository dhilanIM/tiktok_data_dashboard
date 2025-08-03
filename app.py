import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go

# Importar utilidades de estilo
from style_utils import load_css, apply_plotly_dark_theme, get_color_palette

# Configuración de la página
st.set_page_config(
    page_title="Tiktok analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cargar estilos CSS
load_css('styles.css')

# Obtener paleta de colores
colors = get_color_palette()

# Título principal con estilo mejorado
st.markdown("<h1 class='main-header'>TikTok Ads Dashboard</h1>", unsafe_allow_html=True)

# Load data
clean_tiktok_df = pd.read_csv('clean_tiktok_df.csv')

# Sidebar para navegación
st.sidebar.title("🎯 Navigation Menu")
menu_option = st.sidebar.selectbox(
    "Select an option:",
    ["📈 Dashboard", "📉 Linear Regression"]
)

# Función para Dashboard
def dashboard():
    st.header("📈 Analytics Dashboard")
    st.subheader("📊 Key Performance Metrics")    
    
    # Métricas 
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_cost = clean_tiktok_df['Cost'].mean()
        avg_cost = f"${avg_cost:,.2f}"
        st.metric(
            label="💰 Average Ad Cost",
            value=avg_cost,
            delta="12%"
        )
    
    with col2:
        avg_impressions = clean_tiktok_df['Impressions'].mean()
        avg_impressions = f"{avg_impressions:,.2f}"
        st.metric(
            label="👁️ Average Impressions",
            value=avg_impressions,
            delta="8%"
        )
    
    with col3:
        avg_conversions = clean_tiktok_df["Conversions"].mean()
        avg_conversions = f"{avg_conversions:,.2f}"
        st.metric(
            label="🎯 Average Conversions",
            value=avg_conversions,
            delta="-2%"
        )
        
    st.markdown("---")
    
    # Gráficos de ejemplo
    st.subheader("📊 Data Visualizations")
    
    # Gráfico de línea - Impressions over time
    fig_line = px.line(
        clean_tiktok_df.sort_values(by="By Day", ascending=True)
        .drop_duplicates(subset=['By Day']).tail(150), 
        x='By Day', 
        y='Impressions', 
        title='📈 Impressions Over Time'
    )
    fig_line = apply_plotly_dark_theme(fig_line)
    st.plotly_chart(fig_line, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        # Top 5 cheapest ads
        cost = clean_tiktok_df.groupby("Ad name")["Cost"].sum().reset_index().sort_values(by="Cost", ascending=True)
        fig_bar = px.bar(cost.head(5), x='Ad name', y='Cost', title='💸 Top 5 Most Cost-Effective Ads')
        fig_bar = apply_plotly_dark_theme(fig_bar)
        fig_bar.update_traces(marker_color=colors['primary'])
        st.plotly_chart(fig_bar, use_container_width=True)
        
    with col2:
        # Top 5 ads with most impressions
        impressions = clean_tiktok_df.groupby("Ad name")["Impressions"].sum().reset_index().sort_values(by="Impressions", ascending=False)
        fig_bar = px.bar(impressions.head(5), x='Ad name', y='Impressions', title='🔥 Top 5 Ads with Most Impressions')
        fig_bar = apply_plotly_dark_theme(fig_bar)
        fig_bar.update_traces(marker_color=colors['secondary'])
        st.plotly_chart(fig_bar, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        # Top 5 ads with most conversions
        conversions = clean_tiktok_df.groupby("Ad name")["Conversions"].sum().reset_index().sort_values(by="Conversions", ascending=False)
        fig_bar = px.bar(conversions.head(5), x='Ad name', y='Conversions', title='🎯 Top 5 Ads with Most Conversions')
        fig_bar = apply_plotly_dark_theme(fig_bar)
        fig_bar.update_traces(marker_color=colors['success'])
        st.plotly_chart(fig_bar, use_container_width=True)

    with col4:
        # Top 5 ads with 50%+ video views
        video_50 = clean_tiktok_df.groupby("Ad name")["Video views at 50%"].sum().reset_index().sort_values(by="Video views at 50%", ascending=False)
        fig_bar = px.bar(video_50.head(5), x='Ad name', y='Video views at 50%', title='▶️ Top 5 Ads with 50%+ Video Views')
        fig_bar = apply_plotly_dark_theme(fig_bar)
        fig_bar.update_traces(marker_color=colors['warning'])
        st.plotly_chart(fig_bar, use_container_width=True)

# Función para Linear Regression
def linear_regression():
    st.header("📉 Linear Regression Analysis")
    normalized_tiktok_df = pd.read_csv('normalized_tiktok_df.csv')
    
    x_col_names = ['Conversions','Cost', 'CPC (destination)', 'CPM',
       'Impressions', 'Clicks (destination)', 'CTR (destination)', 'Reach',
       'Cost per 1,000 people reached', 'Video views at 25%',
       'Video views at 50%', 'Repeat Impact']
    y_col_names = ['Repeat Impact','CPC (destination)', 'Cost', 'CPM',
       'Impressions', 'Clicks (destination)', 'CTR (destination)', 'Reach',
       'Cost per 1,000 people reached', 'Video views at 25%',
       'Video views at 50%', 'Conversions']
    
    col1, col2 = st.columns(2)
    with col1:
        x_col = st.selectbox("📊 Select X Variable:", x_col_names)
    with col2:
        y_col = st.selectbox("🎯 Select Y Variable:", y_col_names)
    
    normalized_tiktok_df = normalized_tiktok_df[[x_col, y_col]].rename(columns={x_col: 'X', y_col: 'y'})
    
    # Mostrar estadísticas
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Data Description")
        st.dataframe(normalized_tiktok_df.describe())
    
    with col2:
        st.subheader("🔗 Correlation Analysis")
        correlation = normalized_tiktok_df['X'].corr(normalized_tiktok_df['y'])
        st.metric("Correlation Coefficient", f"{correlation:.3f}")
    
    # Realizar regresión
    X = normalized_tiktok_df[['X']]
    y = normalized_tiktok_df['y']
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Entrenar modelo
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predicciones
    y_pred = model.predict(X_test)
    
    # Métricas
    from sklearn.metrics import r2_score, mean_squared_error
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📊 R² Score", f"{r2:.3f}")
    with col2:
        st.metric("📉 MSE", f"{mse:.3f}")
    with col3:
        st.metric("📐 Slope", f"{model.coef_[0]:.3f}")
    
    # Gráfico de dispersión con línea de regresión
    fig = px.scatter(normalized_tiktok_df, x='X', y='y', title='🔍 Linear Regression Analysis')
    
    # Agregar línea de regresión
    x_range = np.linspace(normalized_tiktok_df['X'].min(), normalized_tiktok_df['X'].max(), 100)
    y_line = model.predict(x_range.reshape(-1, 1))
    
    fig.add_trace(go.Scatter(x=x_range, y=y_line, mode='lines', 
                            name='Regression Line', line=dict(color=colors['primary'], width=3)))
    
    fig = apply_plotly_dark_theme(fig)
    fig.update_traces(marker=dict(color=colors['secondary'], size=8, opacity=0.7))
    
    st.plotly_chart(fig, use_container_width=True)

# Lógica de navegación
if menu_option == "📈 Dashboard":
    dashboard()
elif menu_option == "📉 Linear Regression":
    linear_regression()
# elif menu_option == "🎨 Top Creatives tik toks":
#     top_creatives()

# Footer mejorado
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='text-align: center; padding: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-top: 1rem;'>
    <p style='color: white; margin: 0; font-weight: 600;'>👤 Linkedin</p>
    <a href='https://www.linkedin.com/in/dhilan-im/' style='color: white; text-decoration: none; font-weight: 700;'>DhilanIM</a>
</div>
""", unsafe_allow_html=True)