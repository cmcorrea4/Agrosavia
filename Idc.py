import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import re
import traceback
from sodapy import Socrata
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Tuple, List
from sklearn.cluster import KMeans
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from openai import OpenAI

# Configuración de la página
st.set_page_config(
    page_title="Agente Pandas + OpenAI",
    page_icon="🤖",
    layout="wide"
)

# Lista de variables para análisis estadístico
VARIABLES_ESTADISTICAS = [
    'ph_agua_suelo',
    'materia_organica',
    'fosforo_bray_ii',
    'azufre_fosfato_monocalcico',
    'acidez_kcl',
    'aluminio_intercambiable',
    'calcio_intercambiable',
    'magnesio_intercambiable',
    'potasio_intercambiable',
    'sodio_intercambiable',
    'capacidad_de_intercambio_cationico',
    'conductividad_electrica',
    'hierro_disponible_olsen',
    'cobre_disponible',
    'manganeso_disponible_olsen',
    'zinc_disponible_olsen',
    'boro_disponible',
    'hierro_disponible_doble_acido',
    'cobre_disponible_doble_acido',
    'manganeso_disponible_doble_acido',
    'zinc_disponible_doble_acido'
]

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def asignar_tipos_datos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Asigna correctamente los tipos de datos a las columnas del DataFrame.
    Columnas numéricas específicas se convierten a float, el resto a string.
    """
    cols_num = [
        "ph_agua_suelo", "materia_organica", "fosforo_bray_ii",
        "azufre_fosfato_monocalcico", "acidez_kcl",
        "aluminio_intercambiable", "calcio_intercambiable",
        "magnesio_intercambiable", "potasio_intercambiable",
        "sodio_intercambiable", "capacidad_de_intercambio_cationico",
        "conductividad_electrica", "hierro_disponible_olsen",
        "cobre_disponible", "manganeso_disponible_olsen",
        "zinc_disponible_olsen", "boro_disponible",
        "hierro_disponible_doble_acido", "cobre_disponible_doble_acido",
        "manganeso_disponible_doble_acido", "zinc_disponible_doble_acido"
    ]
    
    df_typed = df.copy()
    
    # Convertir columnas numéricas
    for col in cols_num:
        if col in df_typed.columns:
            # Reemplazar comas por puntos y convertir a numérico
            if df_typed[col].dtype == 'object':
                df_typed[col] = pd.to_numeric(
                    df_typed[col].astype(str).str.replace(',', '.').str.strip(),
                    errors='coerce'
                )
            else:
                df_typed[col] = pd.to_numeric(df_typed[col], errors='coerce')
    
    # Convertir el resto de columnas a string
    for col in df_typed.columns:
        if col not in cols_num:
            df_typed[col] = df_typed[col].astype(str)
    
    return df_typed

def preparar_dataframe_numerico(df: pd.DataFrame, variables: List[str] = None) -> pd.DataFrame:
    """
    Convierte DataFrame a numérico de forma consistente para todos los análisis.
    Esta función asegura que todos los cálculos usen los mismos datos convertidos.
    """
    # Seleccionar solo las variables especificadas
    if variables is not None and len(variables) > 0:
        vars_disponibles = [v for v in variables if v in df.columns]
        if vars_disponibles:
            df_work = df[vars_disponibles].copy()
        else:
            return pd.DataFrame()
    else:
        df_work = df.copy()
    
    # Convertir todas las columnas a numéricas
    for col in df_work.columns:
        if df_work[col].dtype == 'object':
            df_work[col] = pd.to_numeric(
                df_work[col].astype(str).str.replace(',', '.').str.strip(),
                errors='coerce'
            )
    
    return df_work

# ============================================================================
# FUNCIONES PARA ÍNDICE DE CALIDAD DE DATOS
# ============================================================================

def calcular_completitud(df: pd.DataFrame, variables: List[str] = None) -> Dict:
    """Calcula la completitud de los datos usando DataFrame convertido a numérico"""
    # Usar DataFrame convertido a numérico
    df_work = preparar_dataframe_numerico(df, variables)
    
    if df_work.empty:
        return {
            'score': 0,
            'pct_completo': 0,
            'columnas_problematicas': {},
            'total_nulos': 0,
            'total_valores': 0
        }
    
    total_cells = df_work.shape[0] * df_work.shape[1]
    non_null_cells = df_work.count().sum()
    pct_completo = (non_null_cells / total_cells) * 100 if total_cells > 0 else 0
    
    null_pct_por_col = (df_work.isnull().sum() / len(df_work)) * 100
    columnas_problematicas = null_pct_por_col[null_pct_por_col > 50].to_dict()
    
    score = (pct_completo / 100) * 25
    penalizacion = len(columnas_problematicas) * 2
    score = max(0, score - penalizacion)
    
    return {
        'score': score,
        'pct_completo': pct_completo,
        'columnas_problematicas': columnas_problematicas,
        'total_nulos': int(df_work.isnull().sum().sum()),
        'total_valores': total_cells
    }

def calcular_unicidad(df: pd.DataFrame, variables: List[str] = None) -> Dict:
    """Calcula la unicidad de los datos usando DataFrame convertido a numérico"""
    # Usar DataFrame convertido a numérico
    df_work = preparar_dataframe_numerico(df, variables)
    
    if df_work.empty:
        return {
            'score': 15,
            'pct_registros_unicos': 100,
            'filas_duplicadas': 0,
            'columnas_con_duplicados_altos': {}
        }
    
    filas_duplicadas = df_work.duplicated().sum()
    total_filas = len(df_work)
    pct_registros_unicos = ((total_filas - filas_duplicadas) / total_filas) * 100 if total_filas > 0 else 100
    
    columnas_con_duplicados_altos = {}
    for col in df_work.columns:
        valores_unicos = df_work[col].nunique()
        valores_totales = df_work[col].notna().sum()
        
        if valores_totales > 0:
            pct_unicos = (valores_unicos / valores_totales) * 100
            if pct_unicos < 20:
                columnas_con_duplicados_altos[col] = {
                    'valores_unicos': valores_unicos,
                    'pct_unicidad': pct_unicos
                }
    
    score = (pct_registros_unicos / 100) * 15
    penalizacion = len(columnas_con_duplicados_altos) * 1
    score = max(0, score - penalizacion)
    
    return {
        'score': score,
        'pct_registros_unicos': pct_registros_unicos,
        'filas_duplicadas': int(filas_duplicadas),
        'columnas_con_duplicados_altos': columnas_con_duplicados_altos
    }

def calcular_consistencia(df: pd.DataFrame, variables: List[str] = None) -> Dict:
    """Calcula la consistencia de los datos usando DataFrame convertido a numérico"""
    # Usar DataFrame convertido a numérico
    df_work = preparar_dataframe_numerico(df, variables)
    
    if df_work.empty:
        return {
            'score': 15,
            'columnas_inconsistentes': {},
            'tipos_mezclados': 0
        }
    
    columnas_inconsistentes = {}
    tipos_mezclados = 0
    
    for col in df_work.columns:
        valores_no_nulos = df_work[col].dropna()
        if len(valores_no_nulos) > 0:
            # Ya está todo convertido a numérico, verificar valores infinitos
            valores_inf = np.isinf(valores_no_nulos).sum()
            if valores_inf > 0:
                columnas_inconsistentes[col] = {
                    'valores_infinitos': int(valores_inf)
                }
                tipos_mezclados += 1
    
    max_score = 15
    if tipos_mezclados > 0:
        score = max_score * (1 - (tipos_mezclados / len(df_work.columns)))
    else:
        score = max_score
    
    return {
        'score': score,
        'columnas_inconsistentes': columnas_inconsistentes,
        'tipos_mezclados': tipos_mezclados
    }

def calcular_precision(df: pd.DataFrame, variables: List[str] = None) -> Dict:
    """Calcula la precisión de los datos usando DataFrame convertido a numérico"""
    # Usar DataFrame convertido a numérico
    df_work = preparar_dataframe_numerico(df, variables)
    
    if df_work.empty:
        return {
            'score': 15,
            'columnas_con_baja_precision': {},
            'precision_promedio': 0
        }
    
    columnas_con_baja_precision = {}
    precisiones = []
    
    for col in df_work.columns:
        valores_no_nulos = df_work[col].dropna()
        if len(valores_no_nulos) > 0:
            # Calcular decimales promedio
            valores_str = valores_no_nulos.astype(str)
            decimales = []
            for val in valores_str:
                if '.' in val:
                    decimales.append(len(val.split('.')[1]))
                else:
                    decimales.append(0)
            
            if decimales:
                decimales_promedio = np.mean(decimales)
                precisiones.append(decimales_promedio)
                
                # Penalizar si hay muy poca precisión (menos de 1 decimal en promedio)
                if decimales_promedio < 1:
                    columnas_con_baja_precision[col] = {
                        'decimales_promedio': decimales_promedio
                    }
    
    precision_promedio = np.mean(precisiones) if precisiones else 0
    
    # Score basado en precisión promedio
    score = min(15, (precision_promedio / 3) * 15)
    penalizacion = len(columnas_con_baja_precision) * 1
    score = max(0, score - penalizacion)
    
    return {
        'score': score,
        'columnas_con_baja_precision': columnas_con_baja_precision,
        'precision_promedio': precision_promedio
    }

def calcular_variabilidad(df: pd.DataFrame, variables: List[str] = None) -> Dict:
    """Calcula la variabilidad de los datos usando DataFrame convertido a numérico"""
    # Usar DataFrame convertido a numérico
    df_work = preparar_dataframe_numerico(df, variables)
    
    if df_work.empty:
        return {
            'score': 15,
            'columnas_sin_variabilidad': {},
            'coeficientes_variacion': {}
        }
    
    columnas_sin_variabilidad = {}
    coeficientes_variacion = {}
    
    for col in df_work.columns:
        valores_no_nulos = df_work[col].dropna()
        if len(valores_no_nulos) > 1:
            # Calcular coeficiente de variación
            media = valores_no_nulos.mean()
            std = valores_no_nulos.std()
            
            if media != 0:
                cv = (std / abs(media)) * 100
                coeficientes_variacion[col] = cv
                
                # Penalizar si no hay variabilidad
                if cv < 0.1:
                    columnas_sin_variabilidad[col] = {
                        'coeficiente_variacion': cv,
                        'valores_unicos': int(valores_no_nulos.nunique())
                    }
    
    score = 15
    penalizacion = len(columnas_sin_variabilidad) * 2
    score = max(0, score - penalizacion)
    
    return {
        'score': score,
        'columnas_sin_variabilidad': columnas_sin_variabilidad,
        'coeficientes_variacion': coeficientes_variacion
    }

def calcular_integridad(df: pd.DataFrame, variables: List[str] = None) -> Dict:
    """Calcula la integridad de los datos usando DataFrame convertido a numérico"""
    # Usar DataFrame convertido a numérico
    df_work = preparar_dataframe_numerico(df, variables)
    
    if df_work.empty:
        return {
            'score': 15,
            'columnas_fuera_rango': {},
            'valores_negativos_inesperados': {}
        }
    
    columnas_fuera_rango = {}
    valores_negativos_inesperados = {}
    
    for col in df_work.columns:
        valores_no_nulos = df_work[col].dropna()
        if len(valores_no_nulos) > 0:
            # Verificar valores extremos usando IQR
            Q1 = valores_no_nulos.quantile(0.25)
            Q3 = valores_no_nulos.quantile(0.75)
            IQR = Q3 - Q1
            
            if IQR > 0:
                limite_inferior = Q1 - 3 * IQR
                limite_superior = Q3 + 3 * IQR
                
                valores_fuera = ((valores_no_nulos < limite_inferior) | 
                               (valores_no_nulos > limite_superior)).sum()
                
                if valores_fuera > 0:
                    pct_fuera = (valores_fuera / len(valores_no_nulos)) * 100
                    if pct_fuera > 5:
                        columnas_fuera_rango[col] = {
                            'valores_fuera_rango': int(valores_fuera),
                            'pct_fuera_rango': pct_fuera
                        }
            
            # Verificar valores negativos si se esperan positivos
            valores_negativos = (valores_no_nulos < 0).sum()
            if valores_negativos > 0:
                pct_negativos = (valores_negativos / len(valores_no_nulos)) * 100
                if pct_negativos > 1:
                    valores_negativos_inesperados[col] = {
                        'valores_negativos': int(valores_negativos),
                        'pct_negativos': pct_negativos
                    }
    
    score = 15
    penalizacion = (len(columnas_fuera_rango) + len(valores_negativos_inesperados)) * 1.5
    score = max(0, score - penalizacion)
    
    return {
        'score': score,
        'columnas_fuera_rango': columnas_fuera_rango,
        'valores_negativos_inesperados': valores_negativos_inesperados
    }

def calcular_icd(df: pd.DataFrame, variables: List[str] = None) -> Tuple[float, Dict]:
    """
    Calcula el Índice de Calidad de Datos (ICD) consolidado.
    Todas las funciones usan la misma conversión numérica centralizada.
    """
    # Calcular cada dimensión
    completitud = calcular_completitud(df, variables)
    unicidad = calcular_unicidad(df, variables)
    consistencia = calcular_consistencia(df, variables)
    precision = calcular_precision(df, variables)
    variabilidad = calcular_variabilidad(df, variables)
    integridad = calcular_integridad(df, variables)
    
    # Sumar scores
    icd_total = (
        completitud['score'] +
        unicidad['score'] +
        consistencia['score'] +
        precision['score'] +
        variabilidad['score'] +
        integridad['score']
    )
    
    # Detalles por dimensión
    detalles = {
        'completitud': completitud,
        'unicidad': unicidad,
        'consistencia': consistencia,
        'precision': precision,
        'variabilidad': variabilidad,
        'integridad': integridad
    }
    
    return icd_total, detalles

# ============================================================================
# FUNCIONES DE DETECCIÓN DE OUTLIERS
# ============================================================================

def detectar_outliers_kmeans(df: pd.DataFrame, variable: str, n_clusters: int = 3) -> pd.DataFrame:
    """Detecta outliers usando K-means clustering"""
    df_work = preparar_dataframe_numerico(df, [variable])
    
    if df_work.empty or variable not in df_work.columns:
        return pd.DataFrame()
    
    # Obtener valores válidos
    valores_validos = df_work[variable].dropna()
    if len(valores_validos) < n_clusters:
        return pd.DataFrame()
    
    # Preparar datos
    X = valores_validos.values.reshape(-1, 1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    # Calcular distancias al centroide más cercano
    distances = np.min(kmeans.transform(X_scaled), axis=1)
    threshold = np.percentile(distances, 95)
    
    # Identificar outliers
    outliers_mask = distances > threshold
    outliers_idx = valores_validos.index[outliers_mask]
    
    # Retornar filas completas
    return df.loc[outliers_idx]

def detectar_outliers_svm(df: pd.DataFrame, variable: str, nu: float = 0.05) -> pd.DataFrame:
    """Detecta outliers usando One-Class SVM"""
    df_work = preparar_dataframe_numerico(df, [variable])
    
    if df_work.empty or variable not in df_work.columns:
        return pd.DataFrame()
    
    # Obtener valores válidos
    valores_validos = df_work[variable].dropna()
    if len(valores_validos) < 10:
        return pd.DataFrame()
    
    # Preparar datos
    X = valores_validos.values.reshape(-1, 1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # One-Class SVM
    svm = OneClassSVM(nu=nu, kernel='rbf', gamma='auto')
    predictions = svm.fit_predict(X_scaled)
    
    # Identificar outliers (predicciones = -1)
    outliers_mask = predictions == -1
    outliers_idx = valores_validos.index[outliers_mask]
    
    # Retornar filas completas
    return df.loc[outliers_idx]

def detectar_outliers_iqr(df: pd.DataFrame, variable: str, factor: float = 1.5) -> pd.DataFrame:
    """Detecta outliers usando el método IQR"""
    df_work = preparar_dataframe_numerico(df, [variable])
    
    if df_work.empty or variable not in df_work.columns:
        return pd.DataFrame()
    
    # Obtener valores válidos
    valores_validos = df_work[variable].dropna()
    if len(valores_validos) < 4:
        return pd.DataFrame()
    
    # Calcular IQR
    Q1 = valores_validos.quantile(0.25)
    Q3 = valores_validos.quantile(0.75)
    IQR = Q3 - Q1
    
    # Límites
    limite_inferior = Q1 - factor * IQR
    limite_superior = Q3 + factor * IQR
    
    # Identificar outliers
    outliers_mask = (valores_validos < limite_inferior) | (valores_validos > limite_superior)
    outliers_idx = valores_validos.index[outliers_mask]
    
    # Retornar filas completas
    return df.loc[outliers_idx]

# ============================================================================
# FUNCIONES DE VISUALIZACIÓN
# ============================================================================

def crear_histogramas(df: pd.DataFrame, variables: List[str]) -> go.Figure:
    """Crea histogramas para las variables seleccionadas"""
    df_work = preparar_dataframe_numerico(df, variables)
    
    if df_work.empty:
        return None
    
    n_vars = len(df_work.columns)
    if n_vars == 0:
        return None
    
    # Determinar layout de subplots
    cols = min(3, n_vars)
    rows = (n_vars + cols - 1) // cols
    
    fig = make_subplots(
        rows=rows, 
        cols=cols,
        subplot_titles=df_work.columns,
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    for idx, col in enumerate(df_work.columns):
        row = idx // cols + 1
        col_pos = idx % cols + 1
        
        valores = df_work[col].dropna()
        if len(valores) > 0:
            fig.add_trace(
                go.Histogram(
                    x=valores,
                    name=col,
                    showlegend=False,
                    marker_color='#636EFA'
                ),
                row=row,
                col=col_pos
            )
    
    fig.update_layout(
        height=300 * rows,
        showlegend=False,
        title_text="Distribución de Variables"
    )
    
    return fig

def crear_boxplots(df: pd.DataFrame, variables: List[str]) -> go.Figure:
    """Crea boxplots para las variables seleccionadas"""
    df_work = preparar_dataframe_numerico(df, variables)
    
    if df_work.empty:
        return None
    
    n_vars = len(df_work.columns)
    if n_vars == 0:
        return None
    
    # Determinar layout de subplots
    cols = min(3, n_vars)
    rows = (n_vars + cols - 1) // cols
    
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=df_work.columns,
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    for idx, col in enumerate(df_work.columns):
        row = idx // cols + 1
        col_pos = idx % cols + 1
        
        valores = df_work[col].dropna()
        if len(valores) > 0:
            fig.add_trace(
                go.Box(
                    y=valores,
                    name=col,
                    showlegend=False,
                    marker_color='#EF553B'
                ),
                row=row,
                col=col_pos
            )
    
    fig.update_layout(
        height=300 * rows,
        showlegend=False,
        title_text="Detección de Valores Atípicos"
    )
    
    return fig

def crear_matriz_correlacion(df: pd.DataFrame, variables: List[str]) -> go.Figure:
    """Crea matriz de correlación"""
    df_work = preparar_dataframe_numerico(df, variables)
    
    if df_work.empty or len(df_work.columns) < 2:
        return None
    
    corr_matrix = df_work.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Correlación")
    ))
    
    fig.update_layout(
        title="Matriz de Correlación",
        xaxis_title="Variables",
        yaxis_title="Variables",
        height=600
    )
    
    return fig

# ============================================================================
# AGENTE OPENAI + PANDAS
# ============================================================================

class OpenAIPandasAgent:
    def __init__(self, model: str, df: pd.DataFrame, api_key: str):
        self.model = model
        self.df = df
        self.client = OpenAI(api_key=api_key)
        self.conversation_history = []
    
    def get_df_info(self) -> str:
        """Obtiene información del DataFrame"""
        info = f"DataFrame con {len(self.df)} filas y {len(self.df.columns)} columnas.\n"
        info += f"Columnas: {', '.join(self.df.columns)}\n"
        info += f"Primeras filas:\n{self.df.head(3).to_string()}\n"
        info += f"Tipos de datos:\n{self.df.dtypes.to_string()}\n"
        info += f"Estadísticas descriptivas:\n{self.df.describe().to_string()}"
        return info
    
    def create_system_prompt(self) -> str:
        """Crea el prompt del sistema"""
        return f"""Eres un asistente experto en análisis de datos con Python y Pandas.
Tienes acceso a un DataFrame con la siguiente información:

{self.get_df_info()}

Tu trabajo es:
1. Analizar las preguntas del usuario sobre este DataFrame
2. Generar código Python/Pandas para responder
3. Ejecutar el código y explicar los resultados

IMPORTANTE:
- El DataFrame está en la variable 'df'
- Genera código Python válido y ejecutable
- Usa pandas, numpy, y librerías estándar
- Responde en español de forma clara y concisa
- Si generas código, enciérralo entre ```python y ```
- Explica los resultados de forma comprensible

Ejemplo de respuesta:
```python
# Código para análisis
resultado = df['columna'].mean()
print(f"La media es: {{resultado}}")
```

Explicación: El código calcula la media de la columna especificada..."""
    
    def extract_code(self, text: str) -> str:
        """Extrae código Python del texto"""
        pattern = r'```python\n(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return matches[0].strip()
        return None
    
    def execute_code(self, code: str) -> Tuple[bool, str]:
        """Ejecuta código Python de forma segura"""
        try:
            # Crear un namespace seguro
            namespace = {
                'df': self.df,
                'pd': pd,
                'np': np,
                'px': px,
                'go': go
            }
            
            # Capturar output
            from io import StringIO
            import sys
            old_stdout = sys.stdout
            sys.stdout = captured_output = StringIO()
            
            # Ejecutar código
            exec(code, namespace)
            
            # Restaurar stdout
            sys.stdout = old_stdout
            output = captured_output.getvalue()
            
            return True, output if output else "Código ejecutado exitosamente"
            
        except Exception as e:
            return False, f"Error: {str(e)}\n{traceback.format_exc()}"
    
    def run_stream(self, question: str):
        """Procesa pregunta con streaming"""
        # Agregar pregunta al historial
        self.conversation_history.append({
            "role": "user",
            "content": question
        })
        
        # Crear mensajes para OpenAI
        messages = [
            {"role": "system", "content": self.create_system_prompt()}
        ] + self.conversation_history
        
        # Contenedor para respuesta
        response_container = st.empty()
        code_container = st.empty()
        result_container = st.empty()
        
        full_response = ""
        
        try:
            # Hacer streaming
            with st.spinner("🤔 Pensando..."):
                stream = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    stream=True,
                    temperature=0.7,
                    max_tokens=2000
                )
                
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        full_response += chunk.choices[0].delta.content
                        response_container.markdown(full_response)
            
            # Agregar respuesta al historial
            self.conversation_history.append({
                "role": "assistant",
                "content": full_response
            })
            
            # Extraer y ejecutar código si existe
            code = self.extract_code(full_response)
            if code:
                with code_container.expander("📝 Ver código generado", expanded=False):
                    st.code(code, language="python")
                
                with st.spinner("⚙️ Ejecutando código..."):
                    success, output = self.execute_code(code)
                    
                    if success:
                        result_container.success("✅ Código ejecutado exitosamente")
                        if output:
                            st.text(output)
                    else:
                        result_container.error("❌ Error al ejecutar código")
                        st.code(output, language="text")
        
        except Exception as e:
            st.error(f"❌ Error al procesar pregunta: {str(e)}")
            st.code(traceback.format_exc(), language="text")

# ============================================================================
# FUNCIONES DE CARGA DE DATOS
# ============================================================================

def cargar_desde_socrata(dominio: str, dataset_id: str, limit: int = 50000) -> pd.DataFrame:
    """Carga datos desde Socrata API"""
    try:
        client = Socrata(dominio, None)
        results = client.get(dataset_id, limit=limit)
        df = pd.DataFrame.from_records(results)
        return df
    except Exception as e:
        st.error(f"Error al cargar datos: {str(e)}")
        return None

def limpiar_datos(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica limpieza básica al DataFrame"""
    df_clean = df.copy()
    
    # Eliminar columnas completamente vacías
    df_clean = df_clean.dropna(axis=1, how='all')
    
    # Eliminar filas completamente vacías
    df_clean = df_clean.dropna(axis=0, how='all')
    
    # Eliminar duplicados exactos
    df_clean = df_clean.drop_duplicates()
    
    # Resetear índice
    df_clean = df_clean.reset_index(drop=True)
    
    return df_clean

# ============================================================================
# ESTADO DE LA APLICACIÓN
# ============================================================================

if 'df' not in st.session_state:
    st.session_state.df = None
if 'df_original' not in st.session_state:
    st.session_state.df_original = None
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'suggested_question' not in st.session_state:
    st.session_state.suggested_question = ""

# ============================================================================
# SIDEBAR
# ============================================================================

st.sidebar.title("⚙️ Configuración")

# Configuración de OpenAI
st.sidebar.subheader("🤖 OpenAI")
api_key = st.sidebar.text_input(
    "API Key",
    type="password",
    help="Ingresa tu API Key de OpenAI"
)

# Modelos disponibles
models_openai = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
    "gpt-3.5-turbo"
]

selected_model = st.sidebar.selectbox(
    "Modelo",
    options=models_openai,
    index=0,
    help="Selecciona el modelo de OpenAI a usar"
)

if not api_key:
    st.sidebar.warning("⚠️ Ingresa tu API Key de OpenAI")

st.sidebar.divider()

# Información del DataFrame
if st.session_state.df is not None:
    st.sidebar.subheader("📊 Info del Dataset")
    st.sidebar.metric("Filas", len(st.session_state.df))
    st.sidebar.metric("Columnas", len(st.session_state.df.columns))
    
    memory_usage = st.session_state.df.memory_usage(deep=True).sum() / 1024**2
    st.sidebar.metric("Memoria", f"{memory_usage:.2f} MB")

# ============================================================================
# INTERFAZ PRINCIPAL
# ============================================================================

st.title("🤖 Agente Pandas + OpenAI")
st.markdown("*Análisis de datos con IA usando la API de OpenAI*")

# Tabs principales
tab1, tab2, tab3 = st.tabs(["📁 Cargar Datos", "📊 Índice de Calidad", "💬 Análisis con IA"])

# TAB 1: CARGAR DATOS
with tab1:
    st.header("📁 Cargar Datos")
    
    # Opciones de carga
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📄 Desde Archivo CSV")
        uploaded_file = st.file_uploader("Sube tu archivo CSV", type=['csv'])
        
        if uploaded_file:
            try:
                df_temp = pd.read_csv(uploaded_file)
                st.success(f"✅ Archivo cargado: {len(df_temp)} filas, {len(df_temp.columns)} columnas")
                
                if st.button("✅ Usar este archivo", type="primary", key="use_csv"):
                    st.session_state.df_original = df_temp.copy()
                    st.session_state.df = asignar_tipos_datos(df_temp)
                    st.session_state.data_loaded = True
                    st.session_state.agent = None
                    st.rerun()
            except Exception as e:
                st.error(f"❌ Error al leer archivo: {str(e)}")
    
    with col2:
        st.subheader("🌍 Desde API Socrata")
        dominio = st.text_input("Dominio", value="www.datos.gov.co")
        dataset_id = st.text_input("Dataset ID", value="ch4u-f3i5")
        limit = st.number_input("Límite de registros", min_value=100, max_value=100000, value=50000, step=1000)
        
        if st.button("🔄 Cargar desde API", type="primary", key="load_socrata"):
            with st.spinner("Cargando datos..."):
                df_temp = cargar_desde_socrata(dominio, dataset_id, limit)
                if df_temp is not None:
                    st.success(f"✅ Datos cargados: {len(df_temp)} filas, {len(df_temp.columns)} columnas")
                    st.session_state.df_original = df_temp.copy()
                    st.session_state.df = asignar_tipos_datos(df_temp)
                    st.session_state.data_loaded = True
                    st.session_state.agent = None
                    st.rerun()
    
    # Preview y limpieza
    if st.session_state.df is not None:
        st.divider()
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader("👀 Vista Previa de Datos")
        with col2:
            if st.button("🧹 Limpiar Datos", use_container_width=True):
                with st.spinner("Limpiando datos..."):
                    df_clean = limpiar_datos(st.session_state.df)
                    filas_antes = len(st.session_state.df)
                    filas_despues = len(df_clean)
                    cols_antes = len(st.session_state.df.columns)
                    cols_despues = len(df_clean.columns)
                    
                    st.session_state.df_original = df_clean.copy()
                    st.session_state.df = asignar_tipos_datos(df_clean)
                    st.session_state.agent = None
                    
                    st.success(f"""
                    ✅ Limpieza completada:
                    - Filas: {filas_antes} → {filas_despues} ({filas_antes - filas_despues} eliminadas)
                    - Columnas: {cols_antes} → {cols_despues} ({cols_antes - cols_despues} eliminadas)
                    """)
                    st.rerun()
        
        st.dataframe(st.session_state.df.head(10), use_container_width=True)
        
        with st.expander("ℹ️ Información del Dataset"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total de Filas", len(st.session_state.df))
            with col2:
                st.metric("Total de Columnas", len(st.session_state.df.columns))
            with col3:
                memory_mb = st.session_state.df.memory_usage(deep=True).sum() / 1024**2
                st.metric("Memoria Usada", f"{memory_mb:.2f} MB")
            
            st.subheader("Columnas y Tipos")
            st.dataframe(
                pd.DataFrame({
                    'Columna': st.session_state.df.columns,
                    'Tipo': st.session_state.df.dtypes.astype(str)
                }),
                hide_index=True,
                use_container_width=True
            )

# TAB 2: ÍNDICE DE CALIDAD
with tab2:
    st.header("📊 Índice de Calidad de Datos (ICD)")
    
    if st.session_state.df_original is not None:
        # Verificar si las variables de análisis existen
        vars_disponibles = [v for v in VARIABLES_ESTADISTICAS if v in st.session_state.df_original.columns]
        
        if vars_disponibles:
            # Selector de variables
            st.subheader("🎯 Selección de Variables")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                variables_seleccionadas = st.multiselect(
                    "Selecciona las variables a analizar:",
                    options=vars_disponibles,
                    default=vars_disponibles[:5] if len(vars_disponibles) >= 5 else vars_disponibles,
                    help="Selecciona las variables numéricas que deseas incluir en el análisis"
                )
            
            with col2:
                st.markdown("###")
                analizar_btn = st.button("🔍 Analizar Calidad", type="primary", use_container_width=True)
            
            if variables_seleccionadas and analizar_btn:
                with st.spinner("📊 Calculando índices de calidad..."):
                    # Calcular ICD
                    icd_total, detalles = calcular_icd(st.session_state.df_original, variables_seleccionadas)
                    
                    if icd_total is not None:
                        # SECCIÓN 1: Métrica principal
                        st.markdown("---")
                        st.subheader("🎯 Índice General")
                        
                        col1, col2, col3 = st.columns([1, 2, 1])
                        with col2:
                            # Determinar color y emoji según el score
                            if icd_total >= 80:
                                color = "green"
                                emoji = "🟢"
                                calificacion = "Excelente"
                            elif icd_total >= 70:
                                color = "blue"
                                emoji = "🔵"
                                calificacion = "Buena"
                            elif icd_total >= 60:
                                color = "orange"
                                emoji = "🟠"
                                calificacion = "Aceptable"
                            elif icd_total >= 40:
                                color = "orange"
                                emoji = "🟠"
                                calificacion = "Baja"
                            else:
                                color = "red"
                                emoji = "🔴"
                                calificacion = "Crítica"
                            
                            st.markdown(f"""
                            <div style='text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 10px;'>
                                <h1 style='color: {color}; font-size: 72px; margin: 0;'>{icd_total:.1f}</h1>
                                <h3 style='color: #666; margin: 10px 0;'>{emoji} Calidad {calificacion}</h3>
                                <p style='color: #888; margin: 0;'>de 100 puntos posibles</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        st.markdown("---")
                        
                        # SECCIÓN 2: Desglose por dimensiones
                        st.subheader("📋 Desglose por Dimensiones")
                        
                        # Crear DataFrame de dimensiones
                        dimensiones_data = []
                        for dim_name, dim_data in detalles.items():
                            score = dim_data['score']
                            max_score = 25 if dim_name == 'completitud' else 15
                            pct = (score / max_score) * 100
                            
                            dimensiones_data.append({
                                'Dimensión': dim_name.capitalize(),
                                'Score': f"{score:.1f}/{max_score}",
                                'Porcentaje': f"{pct:.1f}%"
                            })
                        
                        df_dimensiones = pd.DataFrame(dimensiones_data)
                        st.dataframe(df_dimensiones, hide_index=True, use_container_width=True)
                        
                        # Detalles expandibles de cada dimensión
                        with st.expander("📊 Ver detalles de Completitud"):
                            comp = detalles['completitud']
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("% Completo", f"{comp['pct_completo']:.1f}%")
                            with col2:
                                st.metric("Valores Nulos", comp['total_nulos'])
                            with col3:
                                st.metric("Total Valores", comp['total_valores'])
                            
                            if comp['columnas_problematicas']:
                                st.warning(f"⚠️ {len(comp['columnas_problematicas'])} columnas con >50% nulos")
                                st.json(comp['columnas_problematicas'])
                        
                        with st.expander("📊 Ver detalles de Unicidad"):
                            uni = detalles['unicidad']
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("% Registros Únicos", f"{uni['pct_registros_unicos']:.1f}%")
                            with col2:
                                st.metric("Filas Duplicadas", uni['filas_duplicadas'])
                            
                            if uni['columnas_con_duplicados_altos']:
                                st.warning(f"⚠️ {len(uni['columnas_con_duplicados_altos'])} columnas con baja unicidad")
                                st.json(uni['columnas_con_duplicados_altos'])
                        
                        with st.expander("📊 Ver detalles de Consistencia"):
                            cons = detalles['consistencia']
                            st.metric("Columnas con Tipos Mezclados", cons['tipos_mezclados'])
                            
                            if cons['columnas_inconsistentes']:
                                st.warning(f"⚠️ {len(cons['columnas_inconsistentes'])} columnas con inconsistencias")
                                st.json(cons['columnas_inconsistentes'])
                        
                        with st.expander("📊 Ver detalles de Precisión"):
                            prec = detalles['precision']
                            st.metric("Decimales Promedio", f"{prec['precision_promedio']:.2f}")
                            
                            if prec['columnas_con_baja_precision']:
                                st.warning(f"⚠️ {len(prec['columnas_con_baja_precision'])} columnas con baja precisión")
                                st.json(prec['columnas_con_baja_precision'])
                        
                        with st.expander("📊 Ver detalles de Variabilidad"):
                            var = detalles['variabilidad']
                            
                            if var['columnas_sin_variabilidad']:
                                st.warning(f"⚠️ {len(var['columnas_sin_variabilidad'])} columnas sin variabilidad")
                                st.json(var['columnas_sin_variabilidad'])
                            else:
                                st.success("✅ Todas las columnas tienen variabilidad adecuada")
                            
                            if var['coeficientes_variacion']:
                                st.subheader("Coeficientes de Variación")
                                cv_df = pd.DataFrame([
                                    {'Variable': k, 'CV (%)': f"{v:.2f}"}
                                    for k, v in var['coeficientes_variacion'].items()
                                ])
                                st.dataframe(cv_df, hide_index=True, use_container_width=True)
                        
                        with st.expander("📊 Ver detalles de Integridad"):
                            integ = detalles['integridad']
                            
                            if integ['columnas_fuera_rango']:
                                st.warning(f"⚠️ {len(integ['columnas_fuera_rango'])} columnas con valores fuera de rango")
                                st.json(integ['columnas_fuera_rango'])
                            
                            if integ['valores_negativos_inesperados']:
                                st.warning(f"⚠️ {len(integ['valores_negativos_inesperados'])} columnas con valores negativos inesperados")
                                st.json(integ['valores_negativos_inesperados'])
                            
                            if not integ['columnas_fuera_rango'] and not integ['valores_negativos_inesperados']:
                                st.success("✅ Todos los valores están dentro de rangos esperados")
                        
                        # Interpretación y recomendaciones
                        st.markdown("---")
                        st.subheader("💡 Interpretación y Recomendaciones")
                        
                        if icd_total >= 80:
                            st.success(f"""
                            **{emoji} Calidad excelente ({icd_total:.1f}/100)**
                            
                            Los datos están en óptimas condiciones para análisis. Mínima limpieza requerida.
                            """)
                        elif icd_total >= 70:
                            st.info(f"""
                            **{emoji} Calidad buena ({icd_total:.1f}/100)**
                            
                            Datos confiables con algunos ajustes menores recomendados.
                            """)
                        elif icd_total >= 60:
                            st.warning(f"""
                            **{emoji} Calidad aceptable ({icd_total:.1f}/100)**
                            
                            Se requiere limpieza antes de análisis. Atender recomendaciones.
                            """)
                        elif icd_total >= 40:
                            st.warning(f"""
                            **{emoji} Calidad baja ({icd_total:.1f}/100)**
                            
                            Limpieza profunda requerida. Resultados poco confiables sin procesamiento.
                            """)
                        else:
                            st.error(f"""
                            **{emoji} Calidad crítica ({icd_total:.1f}/100)**
                            
                            Problemas graves. Revisar proceso de captura y validar con fuente original.
                            """)
                        
                        st.divider()
                        
                        # SECCIÓN 3: Visualizaciones
                        st.subheader("📊 Visualizaciones")
                        
                        viz_tab1, viz_tab2, viz_tab3 = st.tabs(["📊 Histogramas", "📦 Boxplots", "🔥 Correlaciones"])
                        
                        with viz_tab1:
                            st.markdown("#### Distribución de Variables")
                            fig_hist = crear_histogramas(st.session_state.df_original, variables_seleccionadas)
                            if fig_hist:
                                st.plotly_chart(fig_hist, use_container_width=True)
                            else:
                                st.warning("No se pudieron generar histogramas")
                        
                        with viz_tab2:
                            st.markdown("#### Detección de Valores Atípicos")
                            fig_box = crear_boxplots(st.session_state.df_original, variables_seleccionadas)
                            if fig_box:
                                st.plotly_chart(fig_box, use_container_width=True)
                            else:
                                st.warning("No se pudieron generar boxplots")
                        
                        with viz_tab3:
                            st.markdown("#### Relaciones entre Variables")
                            if len(variables_seleccionadas) >= 2:
                                fig_corr = crear_matriz_correlacion(st.session_state.df_original, variables_seleccionadas)
                                if fig_corr:
                                    st.plotly_chart(fig_corr, use_container_width=True)
                                    
                                    # Calcular correlaciones para tabla
                                    df_numeric = preparar_dataframe_numerico(st.session_state.df_original, variables_seleccionadas)
                                    
                                    if len(df_numeric.columns) >= 2:
                                        corr_matrix = df_numeric.corr()
                                        
                                        # Extraer correlaciones únicas
                                        correlaciones = []
                                        for i in range(len(corr_matrix.columns)):
                                            for j in range(i+1, len(corr_matrix.columns)):
                                                corr_val = corr_matrix.iloc[i, j]
                                                if not pd.isna(corr_val):
                                                    correlaciones.append({
                                                        'Variable 1': corr_matrix.columns[i],
                                                        'Variable 2': corr_matrix.columns[j],
                                                        'Correlacion': corr_val
                                                    })
                                        
                                        if correlaciones:
                                            df_corr = pd.DataFrame(correlaciones)
                                            df_corr['Correlacion_abs'] = df_corr['Correlacion'].abs()
                                            df_corr = df_corr.sort_values('Correlacion_abs', ascending=False)
                                            
                                            st.markdown("##### Top 10 Correlaciones más Fuertes")
                                            st.dataframe(
                                                df_corr.head(10)[['Variable 1', 'Variable 2', 'Correlacion']].round(3),
                                                hide_index=True,
                                                use_container_width=True
                                            )
                                        else:
                                            st.info("No se pudieron calcular correlaciones válidas")
                                    else:
                                        st.warning("Se necesitan al menos 2 variables numéricas válidas")
                                else:
                                    st.warning("No se pudo generar la matriz de correlación")
                            else:
                                st.info("Selecciona al menos 2 variables para ver correlaciones")
                    
                    else:
                        st.error("❌ No se pudieron calcular estadísticos. Verifica que las variables seleccionadas sean numéricas.")
            
            elif analizar_btn:
                st.warning("⚠️ Por favor selecciona al menos una variable para analizar")
        
        else:
            st.error("❌ No se encontraron las variables especificadas en el dataset")
            st.info("Las columnas disponibles en tu dataset son:")
            st.write(list(st.session_state.df_original.columns))
    
    else:
        st.info("📁 Por favor carga datos primero en la pestaña 'Cargar Datos'")

# TAB 3: ANÁLISIS CON IA
with tab3:
    if st.session_state.df is not None and api_key and selected_model:
        if st.session_state.agent is None:
            st.session_state.agent = OpenAIPandasAgent(selected_model, st.session_state.df, api_key)
        
        st.header("💬 Análisis con IA")
        
        # Usar la pregunta sugerida si existe
        default_question = st.session_state.suggested_question if st.session_state.suggested_question else ""
        
        question = st.text_input(
            "💬 Pregunta sobre tus datos:",
            value=default_question,
            placeholder="Ejemplo: ¿Cuál es la media de la columna ventas?"
        )
        
        # Limpiar la pregunta sugerida después de usarla
        if st.session_state.suggested_question:
            st.session_state.suggested_question = ""
        
        # Preguntas sugeridas
        st.markdown("### 💡 Preguntas Sugeridas")
        st.caption("Haz clic en cualquier pregunta para usarla:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔢 ¿Qué cultivos se dan en cereté?", use_container_width=True, key="q1"):
                st.session_state.suggested_question = "¿Qué cultivos se dan en cereté?"
                st.rerun()
            
            if st.button("📊 ¿Cuál es la desviación estándar del pH de agua?", use_container_width=True, key="q2"):
                st.session_state.suggested_question = "¿Cuál es la desviación estándar del pH de agua?"
                st.rerun()
        
        with col2:
            if st.button("🔍 ¿Cuál es el valor de materia orgánica del café en Liborina?", use_container_width=True, key="q3"):
                st.session_state.suggested_question = "¿Cuál es el valor de materia orgánica del café en Liborina?"
                st.rerun()
            
            if st.button("📈 Muestra los valores atípicos en calcio intercambiable", use_container_width=True, key="q4"):
                st.session_state.suggested_question = "Muestra los valores atípicos en calcio intercambiable"
                st.rerun()
        
        st.divider()
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            analyze_btn = st.button("🔍 Analizar", use_container_width=True, type="primary")
        
        if analyze_btn and question.strip():
            enhanced_question = f"{question}. Responde en español."
            st.session_state.agent.run_stream(enhanced_question)
    else:
        if st.session_state.df is None:
            st.info("📁 Por favor carga datos primero en la pestaña 'Cargar Datos'")
        elif not api_key:
            st.warning("⚠️ Ingresa tu API Key de OpenAI en el sidebar")
        else:
            st.warning("⚠️ Selecciona un modelo de OpenAI en el sidebar")

# Información inicial
if st.session_state.df is None:
    st.divider()
    st.info("""
    ### 🚀 Cómo usar:
    
    **Opción 1: Archivo CSV**
    1. 📁 Sube un archivo CSV
    2. 🧹 Opcional: Aplica limpieza automática
    
    **Opción 2: API Socrata**
    1. 🌍 Ingresa dominio y Dataset ID
    2. 🔄 Carga desde API
    3. 🧹 Opcional: Aplica limpieza automática
    
    **Luego:**
    - 🔑 Ingresa tu API Key de OpenAI
    - 🤖 Selecciona modelo en sidebar
    - 💬 Haz preguntas en lenguaje natural
    - 📊 Consulta Índice de Calidad y estadísticos
    - 📈 Ve visualizaciones y correlaciones
    """)

# Footer
st.divider()
st.caption("🤖 Agente Pandas + OpenAI | Análisis con Índice de Calidad de Datos")
