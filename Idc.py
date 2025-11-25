import streamlit as st
import pandas as pd
import numpy as np
import os
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
import warnings
warnings.filterwarnings('ignore')

def main():
    st.set_page_config(
        page_title="Análisis de Calidad de Datos Agrosavia",
        page_icon="🌱",
        layout="wide"
    )
    
    # Título principal
    st.title("🌱 Sistema de Análisis de Calidad de Datos - Agrosavia")
    st.markdown("**Análisis automatizado de datos de química de suelos con IA**")
    
    # Sidebar para configuración
    with st.sidebar:
        st.header("⚙️ Configuración")
        
        # Campo para API Key de OpenAI
        openai_api_key = st.text_input(
            "🔑 API Key de OpenAI:",
            type="password",
            help="Ingresa tu API key de OpenAI para usar el modelo GPT"
        )
        
        # Selección de modelo
        model_name = st.selectbox(
            "🤖 Modelo OpenAI:",
            ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview"],
            index=0
        )
        
        # Temperatura del modelo
        temperature = st.slider(
            "🌡️ Temperatura:",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.1,
            help="Controla la creatividad de las respuestas"
        )
        
        st.markdown("---")
        st.markdown("### 📊 Sobre el sistema")
        st.markdown("""
        Este sistema analiza:
        - Calidad de datos
        - Valores atípicos
        - Estadísticas descriptivas
        - Consultas con IA
        """)
    
    # Verificar API key
    if not openai_api_key:
        st.warning("⚠️ Por favor, ingresa tu API Key de OpenAI en la barra lateral.")
        st.info("Obtén tu API key en: https://platform.openai.com/api-keys")
        return
    
    os.environ["OPENAI_API_KEY"] = openai_api_key
    
    # Carga de archivo
    st.header("📁 Carga de Datos")
    uploaded_file = st.file_uploader(
        "Selecciona un archivo CSV o Excel:",
        type=['csv', 'xlsx', 'xls'],
        help="Formatos soportados: CSV, Excel"
    )
    
    if uploaded_file is not None:
        try:
            # Leer archivo
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success(f"✅ Archivo cargado: {uploaded_file.name}")
            
            # Métricas básicas
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📏 Filas", f"{df.shape[0]:,}")
            with col2:
                st.metric("📊 Columnas", df.shape[1])
            with col3:
                st.metric("💾 Memoria", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
            with col4:
                st.metric("🔢 Valores Únicos", f"{df.nunique().sum():,}")
            
            # Tabs principales
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📋 Vista de Datos",
                "📊 Calidad de Datos", 
                "🔍 Análisis de Outliers",
                "📈 Estadísticas",
                "🤖 Consultas IA"
            ])
            
            # TAB 1: Vista de Datos
            with tab1:
                st.subheader("Vista Previa de los Datos")
                
                # Filtros
                col1, col2 = st.columns(2)
                with col1:
                    num_rows = st.slider("Número de filas a mostrar:", 10, min(500, len(df)), 100)
                with col2:
                    columns_to_show = st.multiselect(
                        "Selecciona columnas:",
                        options=df.columns.tolist(),
                        default=df.columns.tolist()[:10] if len(df.columns) > 10 else df.columns.tolist()
                    )
                
                if columns_to_show:
                    st.dataframe(df[columns_to_show].head(num_rows), use_container_width=True)
                
                # Información de columnas
                with st.expander("ℹ️ Información de Columnas"):
                    info_df = pd.DataFrame({
                        'Columna': df.columns,
                        'Tipo': df.dtypes.astype(str),
                        'No Nulos': df.count(),
                        'Nulos': df.isnull().sum(),
                        '% Nulos': (df.isnull().sum() / len(df) * 100).round(2),
                        'Únicos': df.nunique()
                    })
                    st.dataframe(info_df, use_container_width=True)
            
            # TAB 2: Calidad de Datos
            with tab2:
                st.subheader("📊 Índice de Calidad de Datos (ICD)")
                
                # Calcular métricas de calidad
                completitud = (1 - df.isnull().sum() / len(df)) * 100
                unicidad = (df.nunique() / len(df)) * 100
                
                # ICD por columna
                icd_data = pd.DataFrame({
                    'Columna': df.columns,
                    'Completitud (%)': completitud.round(2),
                    'Unicidad (%)': unicidad.round(2),
                    'ICD (%)': ((completitud + unicidad) / 2).round(2)
                })
                icd_data = icd_data.sort_values('ICD (%)', ascending=False)
                
                # Mostrar tabla
                st.dataframe(icd_data, use_container_width=True)
                
                # Gráfico de ICD
                st.subheader("Visualización del ICD")
                chart_data = icd_data.set_index('Columna')['ICD (%)']
                st.bar_chart(chart_data)
                
                # Resumen general
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("ICD Promedio", f"{icd_data['ICD (%)'].mean():.2f}%")
                with col2:
                    st.metric("Mejor ICD", f"{icd_data['ICD (%)'].max():.2f}%")
                with col3:
                    st.metric("Peor ICD", f"{icd_data['ICD (%)'].min():.2f}%")
            
            # TAB 3: Análisis de Outliers
            with tab3:
                st.subheader("🔍 Detección de Valores Atípicos")
                
                # Seleccionar columnas numéricas
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                
                if numeric_cols:
                    # Método de detección
                    method = st.selectbox(
                        "Método de detección:",
                        ["IQR (Rango Intercuartílico)", "Z-Score", "Ambos"]
                    )
                    
                    # Columna a analizar
                    selected_col = st.selectbox("Selecciona columna:", numeric_cols)
                    
                    if selected_col:
                        col_data = df[selected_col].dropna()
                        
                        outliers_iqr = pd.Series([False] * len(df))
                        outliers_zscore = pd.Series([False] * len(df))
                        
                        # Detección por IQR
                        if method in ["IQR (Rango Intercuartílico)", "Ambos"]:
                            Q1 = col_data.quantile(0.25)
                            Q3 = col_data.quantile(0.75)
                            IQR = Q3 - Q1
                            lower_bound = Q1 - 1.5 * IQR
                            upper_bound = Q3 + 1.5 * IQR
                            outliers_iqr = (df[selected_col] < lower_bound) | (df[selected_col] > upper_bound)
                        
                        # Detección por Z-Score
                        if method in ["Z-Score", "Ambos"]:
                            z_scores = np.abs((df[selected_col] - col_data.mean()) / col_data.std())
                            outliers_zscore = z_scores > 3
                        
                        # Combinar outliers según método
                        if method == "Ambos":
                            outliers = outliers_iqr | outliers_zscore
                        elif method == "IQR (Rango Intercuartílico)":
                            outliers = outliers_iqr
                        else:
                            outliers = outliers_zscore
                        
                        # Mostrar resultados
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total de Outliers", f"{outliers.sum():,}")
                        with col2:
                            st.metric("Porcentaje", f"{(outliers.sum()/len(df)*100):.2f}%")
                        with col3:
                            st.metric("Valores Normales", f"{(~outliers).sum():,}")
                        
                        # Gráfico de distribución
                        st.subheader("Distribución de Datos")
                        st.line_chart(df[selected_col].value_counts().sort_index())
                        
                        # Mostrar outliers
                        if outliers.sum() > 0:
                            with st.expander(f"Ver {outliers.sum()} registros con outliers"):
                                st.dataframe(df[outliers], use_container_width=True)
                else:
                    st.info("No hay columnas numéricas para analizar outliers.")
            
            # TAB 4: Estadísticas
            with tab4:
                st.subheader("📈 Estadísticas Descriptivas")
                
                numeric_df = df.select_dtypes(include=[np.number])
                
                if not numeric_df.empty:
                    # Estadísticas generales
                    st.dataframe(numeric_df.describe(), use_container_width=True)
                    
                    # Gráficos de distribución
                    st.subheader("Distribuciones por Columna")
                    selected_stat_col = st.selectbox(
                        "Selecciona columna para visualizar:",
                        numeric_df.columns.tolist()
                    )
                    
                    if selected_stat_col:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Histograma**")
                            hist_data = numeric_df[selected_stat_col].dropna()
                            st.bar_chart(hist_data.value_counts().sort_index())
                        
                        with col2:
                            st.write("**Estadísticas**")
                            stats = {
                                'Media': hist_data.mean(),
                                'Mediana': hist_data.median(),
                                'Desv. Est.': hist_data.std(),
                                'Mínimo': hist_data.min(),
                                'Máximo': hist_data.max()
                            }
                            for key, value in stats.items():
                                st.metric(key, f"{value:.2f}")
                else:
                    st.info("No hay columnas numéricas para mostrar estadísticas.")
            
            # TAB 5: Consultas IA
            with tab5:
                st.subheader("🤖 Asistente de Análisis con IA")
                
                try:
                    # Inicializar agente
                    llm = ChatOpenAI(
                        model=model_name,
                        temperature=temperature,
                        openai_api_key=openai_api_key
                    )
                    
                    agent = create_pandas_dataframe_agent(
                        llm,
                        df,
                        verbose=True,
                        agent_type=AgentType.OPENAI_FUNCTIONS,
                        allow_dangerous_code=True
                    )
                    
                    st.success("✅ Agente IA inicializado")
                    
                    # Ejemplos de preguntas
                    with st.expander("💡 Ejemplos de preguntas"):
                        ejemplos = [
                            "¿Cuál es la correlación entre las variables numéricas?",
                            "¿Cuáles son las columnas con más valores nulos?",
                            "Muestra estadísticas de la columna [nombre]",
                            "¿Cuáles son los 10 valores más frecuentes en [columna]?",
                            "¿Hay patrones en los datos faltantes?",
                            "Resume las principales características del dataset"
                        ]
                        for i, ej in enumerate(ejemplos, 1):
                            st.write(f"{i}. {ej}")
                    
                    # Historial de chat
                    if 'chat_history' not in st.session_state:
                        st.session_state.chat_history = []
                    
                    # Input de pregunta
                    user_question = st.text_area(
                        "Escribe tu pregunta:",
                        placeholder="Ej: ¿Cuál es la distribución de la columna pH?",
                        height=100
                    )
                    
                    col1, col2 = st.columns([1, 5])
                    with col1:
                        ask_btn = st.button("🚀 Preguntar", type="primary", use_container_width=True)
                    with col2:
                        clear_btn = st.button("🗑️ Limpiar historial", use_container_width=True)
                    
                    if clear_btn:
                        st.session_state.chat_history = []
                        st.rerun()
                    
                    if ask_btn and user_question:
                        with st.spinner("🔄 Analizando..."):
                            try:
                                response = agent.invoke({"input": user_question})
                                
                                st.session_state.chat_history.append({
                                    "question": user_question,
                                    "answer": response["output"]
                                })
                                
                                st.rerun()
                                
                            except Exception as e:
                                st.error(f"❌ Error: {str(e)}")
                                st.info("💡 Intenta reformular tu pregunta")
                    
                    # Mostrar historial
                    if st.session_state.chat_history:
                        st.markdown("---")
                        st.subheader("💬 Historial de Conversación")
                        
                        for i, chat in enumerate(reversed(st.session_state.chat_history)):
                            with st.expander(
                                f"❓ {chat['question'][:60]}..." if len(chat['question']) > 60 
                                else f"❓ {chat['question']}", 
                                expanded=(i==0)
                            ):
                                st.write("**Pregunta:**")
                                st.info(chat['question'])
                                st.write("**Respuesta:**")
                                st.success(chat['answer'])
                
                except Exception as e:
                    st.error(f"❌ Error al inicializar agente: {str(e)}")
                    st.info("Verifica tu API key de OpenAI")
        
        except Exception as e:
            st.error(f"❌ Error al cargar archivo: {str(e)}")
            st.info("Verifica el formato del archivo")
    
    else:
        st.info("👆 Carga un archivo para comenzar el análisis")
        
        # Información adicional
        st.markdown("---")
        st.subheader("ℹ️ Características del Sistema")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Análisis de Calidad:**
            - Índice de Calidad de Datos (ICD)
            - Completitud y Unicidad
            - Detección de valores nulos
            """)
        
        with col2:
            st.markdown("""
            **Análisis Avanzado:**
            - Detección de outliers (IQR, Z-Score)
            - Estadísticas descriptivas
            - Consultas con IA usando OpenAI GPT
            """)

if __name__ == "__main__":
    main()
