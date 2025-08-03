import streamlit as st
import os

def load_css(file_name):
    try:
        # Verificar si el archivo existe
        if os.path.exists(file_name):
            with open(file_name) as f:
                st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
        else:
            st.warning(f"⚠️ Archivo CSS '{file_name}' no encontrado. Usando estilos por defecto.")
    except Exception as e:
        st.error(f"❌ Error al cargar estilos CSS: {e}")

def apply_plotly_dark_theme(fig):
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#e2e8f0',
        title_font_size=16,
        title_font_color='#e2e8f0'
    )
    fig.update_xaxes(gridcolor='#4a5568')
    fig.update_yaxes(gridcolor='#4a5568')
    
    return fig

def get_color_palette():
    return {
        'primary': '#667eea',
        'secondary': '#764ba2',
        'success': '#48bb78',
        'warning': '#ed8936',
        'danger': '#f56565',
        'info': '#4299e1',
        'light': '#e2e8f0',
        'dark': '#2d3748',
        'background': '#0f1419',
        'surface': '#2d3748'
    }