import streamlit as st
import pandas as pd
import os
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
import warnings
warnings.filterwarnings('ignore')

def main():
    st.set_page_config(
        page_title="Análisis de Datos con IA",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 Análisis de Datos")
    st.markdown("**Carga tu archivo CSV/XLS y haz preguntas sobre tus datos usando IA**")
    
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
            help="Controla la creatividad de las respuestas (0 = más preciso, 1 = más creativo)"
        )
    
    # Verificar si se ha ingresado la API key
    if not openai_api_key:
        st.warning("⚠️ Por favor, ingresa tu API Key de OpenAI en la barra lateral.")
        st.info("Puedes obtener tu API key en: https://platform.openai.com/api-keys")
    
    # Configurar la variable de entorno
    os.environ["OPENAI_API_KEY"] = openai_api_key
    
    # Carga de archivo
    st.header("📁 Carga tu archivo")
    uploaded_file = st.file_uploader(
        "Selecciona un archivo CSV o Excel:",
        type=['csv', 'xlsx', 'xls'],
        help="Formatos soportados: CSV, Excel (.xlsx, .xls)"
    )
    
    if uploaded_file is not None:
        try:
            # Leer el archivo según su tipo
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success(f"✅ Archivo cargado exitosamente: {uploaded_file.name}")
            
            # Mostrar información básica del dataset
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📏 Filas", df.shape[0])
            with col2:
                st.metric("📊 Columnas", df.shape[1])
            with col3:
                st.metric("💾 Tamaño", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
            
            # Vista previa de los datos
            st.header("👀 Vista previa de los datos")
            
            # Tabs para diferentes vistas
            tab1, tab2, tab3 = st.tabs(["📋 Datos", "📈 Información", "🔍 Estadísticas"])
            
            with tab1:
                st.dataframe(df.head(100), use_container_width=True)
                
            with tab2:
                st.subheader("Información del Dataset")
                info_df = pd.DataFrame({
                    'Columna': df.columns,
                    'Tipo': df.dtypes.astype(str),
                    'No Nulos': df.count(),
                    'Nulos': df.isnull().sum(),
                    '% Nulos': (df.isnull().sum() / len(df) * 100).round(2)
                })
                st.dataframe(info_df, use_container_width=True)
                
                # Mostrar información adicional del dataset
                st.write("**Resumen del Dataset:**")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Memoria (MB)", f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f}")
                with col2:
                    st.metric("Columnas Numéricas", len(df.select_dtypes(include=['number']).columns))
                with col3:
                    st.metric("Columnas Texto", len(df.select_dtypes(include=['object']).columns))
                with col4:
                    st.metric("Valores Únicos Totales", df.nunique().sum())
                
            with tab3:
                st.subheader("Estadísticas Descriptivas")
                numeric_df = df.select_dtypes(include=['number'])
                if not numeric_df.empty:
                    st.dataframe(numeric_df.describe(), use_container_width=True)
                else:
                    st.info("No hay columnas numéricas para mostrar estadísticas.")
            
            # Crear el agente de pandas
            st.header("🤖 Agente de Análisis IA")
            
            try:
                # Inicializar el modelo de OpenAI
                llm = ChatOpenAI(
                    model=model_name,
                    temperature=temperature,
                    openai_api_key=openai_api_key
                )
                
                # Crear el agente de pandas
                agent = create_pandas_dataframe_agent(
                    llm,
                    df,
                    verbose=True,
                    agent_type=AgentType.OPENAI_FUNCTIONS,
                    allow_dangerous_code=True  # Necesario para ejecutar código
                )
                
                st.success("🎯 Agente IA inicializado correctamente")
                
                # Ejemplos de preguntas
                st.subheader("💡 Ejemplos de preguntas que puedes hacer:")
                examples = [
                    "¿Cuántas filas tiene el dataset?",
                    "¿Cuáles son las columnas numéricas?",
                    "Muestra un resumen estadístico de los datos",
                    "¿Hay valores nulos en el dataset?",
                    "¿Cuál es la correlación entre las variables numéricas?",
                    "Crea un gráfico de las variables más importantes",
                    "¿Cuáles son los valores únicos de [nombre_columna]?",
                    "Calcula la media, mediana y moda de [columna_numerica]"
                ]
                
                for i, example in enumerate(examples, 1):
                    st.write(f"{i}. {example}")
                
                # Interface para hacer preguntas
                st.subheader("❓ Haz tu pregunta sobre los datos")
                
                # Historial de conversación
                if 'chat_history' not in st.session_state:
                    st.session_state.chat_history = []
                
                # Campo de entrada para la pregunta
                user_question = st.text_input(
                    "Escribe tu pregunta:",
                    placeholder="Ej: ¿Cuál es la correlación entre las variables numéricas?",
                    key="user_input"
                )
                
                col1, col2 = st.columns([1, 4])
                with col1:
                    ask_button = st.button("🚀 Preguntar", type="primary")
                with col2:
                    clear_button = st.button("🗑️ Limpiar historial")
                
                if clear_button:
                    st.session_state.chat_history = []
                    st.rerun()
                
                if ask_button and user_question:
                    with st.spinner("🔄 El agente está analizando tus datos..."):
                        try:
                            # Ejecutar la pregunta con el agente
                            response = agent.invoke({"input": user_question})
                            
                            # Agregar al historial
                            st.session_state.chat_history.append({
                                "question": user_question,
                                "answer": response["output"]
                            })
                            
                            # Limpiar el campo de entrada
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"❌ Error al procesar la pregunta: {str(e)}")
                            st.info("💡 Intenta reformular tu pregunta o verifica que la columna mencionada existe en el dataset.")
                
                # Mostrar historial de conversación
                if st.session_state.chat_history:
                    st.subheader("💬 Historial de conversación")
                    
                    for i, chat in enumerate(reversed(st.session_state.chat_history)):
                        with st.expander(f"❓ {chat['question'][:60]}..." if len(chat['question']) > 60 else f"❓ {chat['question']}", expanded=(i==0)):
                            st.write("**Pregunta:**")
                            st.write(chat['question'])
                            st.write("**Respuesta:**")
                            st.write(chat['answer'])
                            st.divider()
                
            except Exception as e:
                st.error(f"❌ Error al inicializar el agente: {str(e)}")
                st.info("Verifica que tu API key de OpenAI sea válida y tenga créditos disponibles.")
                
        except Exception as e:
            st.error(f"❌ Error al cargar el archivo: {str(e)}")
            st.info("Verifica que el archivo tenga el formato correcto (CSV o Excel).")
    
    else:
        st.info("👆 Carga un archivo CSV o Excel para comenzar el análisis.")
        
        # Información adicional cuando no hay archivo cargado
        st.markdown("---")
        st.subheader("ℹ️ Sobre esta aplicación")
        st.markdown("""
        Esta aplicación utiliza:
        
        - **OpenAI GPT** para el procesamiento de lenguaje natural
        
        **Funcionalidades:**
        - Carga archivos CSV y Excel
        - Análisis automático de datos con IA
        - Respuestas en lenguaje natural
        - Generación de estadísticas y visualizaciones
        - Historial de conversación
        """)

if __name__ == "__main__":
    main()
