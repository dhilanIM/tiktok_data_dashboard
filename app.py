import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go


# Load data
clean_tiktok_df = pd.read_csv('clean_tiktok_df.csv')


# Configuración de la página
st.set_page_config(
    page_title="Tiktok  analysis ",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Título principal
st.title("📊 Tiktok ads analysis")


# Sidebar para navegación
st.sidebar.title("Menu")
menu_option = st.sidebar.selectbox(
    "Select an option:",
    ["Dashboard", "Linear Regression", "Top Creatives tik toks"]
)


# Función para Dashboard
def dashboard():
    st.header("📈 Dashboard")
    st.subheader("Summary")
    
    # Métricas 
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Avg Ad Cost",
            value="1,234",
            delta="12%"
        )
    
    with col2:
        st.metric(
            label="Avg Impressions",
            value="45,678",
            delta="8%"
        )
    
    with col3:
        st.metric(
            label="Avg Conversions",
            value="89",
            delta="-2%"
        )
   
    # Gráficos de ejemplo
    st.subheader("Vizualizations")
    
    fig_line = px.line(clean_tiktok_df.sort_values(by="By Day", 
                                            ascending=True).drop_duplicates(subset=['By Day']).tail(150), 
                                            x='By Day', y='Impressions', title='Impressions Over Time')
    st.plotly_chart(fig_line, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        # Get the total cost of each Ad (lower is better)
        cost = clean_tiktok_df.groupby("Ad name")["Cost"].sum().reset_index().sort_values(by="Cost",
                                                                                          ascending=True)
        fig_bar = px.bar(cost.head(5), x='Ad name', y='Cost', title='Top 5 Cheapest ads (lower is better)')
        st.plotly_chart(fig_bar, use_container_width=True)
        
    with col2:
        # Get the # of impressions per Ad (Higher is better)
        impressions = clean_tiktok_df.groupby("Ad name")["Impressions"].sum().reset_index().sort_values(by="Impressions",
                                    ascending=False)
        fig_bar = px.bar(impressions.head(5), x='Ad name', y='Impressions', title='Top 5 ads with more impressions')
        st.plotly_chart(fig_bar, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        # Get the conversions per Ad  (Higher is better)
        conversions = clean_tiktok_df.groupby("Ad name")["Conversions"].sum().reset_index().sort_values(by="Conversions",
                                    ascending=False)
        fig_bar = px.bar(conversions.head(5), x='Ad name', y='Conversions', title='Top 5 ads with more conversions')
        st.plotly_chart(fig_bar, use_container_width=True)

    with col4:
        # Get the repeat impact ped Ad (Higher is better)
        video_50 = clean_tiktok_df.groupby("Ad name")["Video views at 50%"].sum().reset_index().sort_values(by="Video views at 50%",
                                     ascending=False)
        fig_bar = px.bar(video_50.head(5), x='Ad name', y='Video views at 50%', title='Top 5 ads viewed at 50% or more')
        st.plotly_chart(fig_bar, use_container_width=True)
        
    
# Función para Linear Regression
def linear_regression():
    st.header("📉 Linear Regression")
    normalized_tiktok_df = pd.read_csv('normalized_tiktok_df.csv')
    
    x_col_names = ['Conversions','Cost', 'CPC (destination)', 'CPM',
       'Impressions', 'Clicks (destination)', 'CTR (destination)', 'Reach',
       'Cost per 1,000 people reached', 'Video views at 25%',
       'Video views at 50%', 'Repeat Impact']
    y_col_names = ['Repeat Impact','CPC (destination)', 'Cost', 'CPM',
       'Impressions', 'Clicks (destination)', 'CTR (destination)', 'Reach',
       'Cost per 1,000 people reached', 'Video views at 25%',
       'Video views at 50%', 'Conversions']
    x_col = st.selectbox("Select variable X:", x_col_names)  # Error cuando hay dos columas iguales seleccionadas 
    y_col = st.selectbox("Select variable Y:", y_col_names)
    
    normalized_tiktok_df = normalized_tiktok_df[[x_col, y_col]].rename(columns={x_col: 'X', y_col: 'y'})
    # Mostrar estadísticas
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Data description")
        st.dataframe(normalized_tiktok_df.describe())
    
    with col2:
        st.subheader("Correlation")
        correlation = normalized_tiktok_df['X'].corr(normalized_tiktok_df['y'])
        st.metric("Correlation Coefficients", f"{correlation:.3f}")
    
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
        st.metric("R² Score", f"{r2:.3f}")
    with col2:
        st.metric("MSE", f"{mse:.3f}")
    with col3:
        st.metric("Slope", f"{model.coef_[0]:.3f}")
    
    # Gráfico de dispersión con línea de regresión
    fig = px.scatter(normalized_tiktok_df, x='X', y='y', title='Linear Regression')
    
    # Agregar línea de regresión
    x_range = np.linspace(normalized_tiktok_df['X'].min(), normalized_tiktok_df['X'].max(), 100)
    y_line = model.predict(x_range.reshape(-1, 1))
    
    fig.add_trace(go.Scatter(x=x_range, y=y_line, mode='lines', 
                            name='Linear Regression', line=dict(color='red')))
    
    st.plotly_chart(fig, use_container_width=True)


# Función para Top Creatives
def top_creatives():
    st.header("🎨 Top Creatives")
    
    # Filtros
    col1, col2, col3 = st.columns(3)
    
    with col1:
        date_range = st.date_input("Rango de fechas:", 
                                  value=[pd.to_datetime('2024-01-01'), pd.to_datetime('2024-01-31')])
    
    with col2:
        campaign_filter = st.selectbox("Campaña:", 
                                     ["Todas", "Campaña A", "Campaña B", "Campaña C"])
    
    with col3:
        metric_sort = st.selectbox("Ordenar por:", 
                                 ["CTR", "Conversiones", "CPC", "Impresiones"])
    
    # Datos de ejemplo para creativos
    np.random.seed(42)
    creative_data = {
        'Creative_ID': [f'CR_{i:03d}' for i in range(1, 21)],
        'Nombre': [f'Creative {i}' for i in range(1, 21)],
        'Campaña': np.random.choice(['Campaña A', 'Campaña B', 'Campaña C'], 20),
        'Impresiones': np.random.randint(1000, 50000, 20),
        'Clicks': np.random.randint(10, 1000, 20),
        'Conversiones': np.random.randint(1, 100, 20),
        'Costo': np.random.uniform(100, 5000, 20).round(2)
    }
    
    df_creatives = pd.DataFrame(creative_data)
    
    # Calcular métricas
    df_creatives['CTR'] = (df_creatives['Clicks'] / df_creatives['Impresiones'] * 100).round(2)
    df_creatives['CPC'] = (df_creatives['Costo'] / df_creatives['Clicks']).round(2)
    df_creatives['Conversion_Rate'] = (df_creatives['Conversiones'] / df_creatives['Clicks'] * 100).round(2)
    
    # Aplicar filtros
    if campaign_filter != "Todas":
        df_creatives = df_creatives[df_creatives['Campaña'] == campaign_filter]
    
    # Ordenar por métrica seleccionada
    ascending = False if metric_sort in ['CTR', 'Conversiones'] else True
    df_sorted = df_creatives.sort_values(metric_sort, ascending=ascending)
    
    # Mostrar top 10
    st.subheader(f"Top 10 Creativos - Ordenado por {metric_sort}")
    
    # Tabla con métricas destacadas
    top_10 = df_sorted.head(10)
    
    # Crear gráfico de barras para el top 10
    fig = px.bar(top_10, x='Creative_ID', y=metric_sort, 
                 color='Campaña', title=f'Top 10 Creativos por {metric_sort}')
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)
    
    # Tabla detallada
    st.subheader("Tabla Detallada")
    st.dataframe(
        top_10[['Creative_ID', 'Nombre', 'Campaña', 'Impresiones', 'Clicks', 
                'CTR', 'Conversiones', 'Conversion_Rate', 'CPC', 'Costo']],
        use_container_width=True
    )
    
    # Métricas de resumen
    st.subheader("Resumen General")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Impresiones", f"{df_creatives['Impresiones'].sum():,}")
    
    with col2:
        st.metric("Total Clicks", f"{df_creatives['Clicks'].sum():,}")
    
    with col3:
        st.metric("CTR Promedio", f"{df_creatives['CTR'].mean():.2f}%")
    
    with col4:
        st.metric("Total Conversiones", f"{df_creatives['Conversiones'].sum():,}")

# Lógica de navegación
if menu_option == "Dashboard":
    dashboard()
elif menu_option == "Linear Regression":
    linear_regression()
elif menu_option == "Top Creatives tik toks":
    top_creatives()

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("💻 In/DhilanIM")