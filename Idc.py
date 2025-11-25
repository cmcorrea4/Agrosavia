import streamlit as st
import pandas as pd
import numpy as np
import os
import requests
import json
from sodapy import Socrata
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Tuple, List
from sklearn.cluster import KMeans
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
import warnings
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="Agente datos suelos Agrosavia",
    page_icon="📊",
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
    
    score_registros = (pct_registros_unicos / 100) * 10
    score_columnas = 5  # Score fijo sin penalización
    score = score_registros + score_columnas
    
    return {
        'score': score,
        'pct_registros_unicos': pct_registros_unicos,
        'filas_duplicadas': int(filas_duplicadas),
        'columnas_con_duplicados_altos': columnas_con_duplicados_altos
    }

def calcular_consistencia(df: pd.DataFrame, variables: List[str] = None) -> Dict:
    """Calcula la consistencia de los datos comparando original vs convertido"""
    # Seleccionar variables
    if variables is not None and len(variables) > 0:
        vars_disponibles = [v for v in variables if v in df.columns]
        if vars_disponibles:
            df_original = df[vars_disponibles].copy()
        else:
            return {
                'score': 15,
                'pct_consistente': 100,
                'columnas_tipo_mixto': {},
                'valores_inconsistentes': 0
            }
    else:
        df_original = df.copy()
    
    # Convertir a numérico
    df_convertido = preparar_dataframe_numerico(df, variables)
    
    if df_convertido.empty:
        return {
            'score': 15,
            'pct_consistente': 100,
            'columnas_tipo_mixto': {},
            'valores_inconsistentes': 0
        }
    
    # Contar valores que se perdieron en la conversión
    inconsistencias = 0
    columnas_tipo_mixto = {}
    
    for col in df_convertido.columns:
        # Comparar valores no nulos antes y después
        nulos_original = df_original[col].isnull().sum()
        nulos_convertido = df_convertido[col].isnull().sum()
        valores_perdidos = nulos_convertido - nulos_original
        
        if valores_perdidos > 0:
            columnas_tipo_mixto[col] = int(valores_perdidos)
            inconsistencias += valores_perdidos
    
    total_valores = df_original.shape[0] * df_original.shape[1]
    pct_consistente = ((total_valores - inconsistencias) / total_valores) * 100 if total_valores > 0 else 100
    score = (pct_consistente / 100) * 15
    
    return {
        'score': score,
        'pct_consistente': pct_consistente,
        'columnas_tipo_mixto': columnas_tipo_mixto,
        'valores_inconsistentes': int(inconsistencias)
    }

def calcular_precision_outliers(df: pd.DataFrame, variables_numericas: List[str] = None, metodo: str = 'iqr') -> Dict:
    """
    Calcula la precisión basada en detección de outliers
    
    Args:
        df: DataFrame con los datos
        variables_numericas: Lista de variables a analizar
        metodo: 'iqr', 'kmeans', 'svm', o 'combinado'
    """
    # Usar DataFrame convertido a numérico
    df_numeric = preparar_dataframe_numerico(df, variables_numericas)
    
    if df_numeric.empty:
        return {
            'score': 20,
            'pct_datos_precisos': 100,
            'outliers_por_columna': {},
            'total_outliers': 0,
            'total_datos_numericos': 0,
            'metodo_usado': metodo,
            'df_outliers_completo': pd.DataFrame(),
            'num_filas_con_outliers': 0
        }
    
    outliers_por_columna = {}
    total_outliers = 0
    total_datos = 0
    todos_indices_outliers = set()  # Para recopilar todos los índices únicos de outliers
    
    for col in df_numeric.columns:
        data = df_numeric[col].dropna()
        if len(data) == 0:
            continue
            
        n_outliers = 0
        outlier_info = {'variable': col}
        
        # Inicializar contadores y conjuntos de índices para cada método
        n_outliers_iqr = 0
        n_outliers_kmeans = 0
        n_outliers_svm = 0
        indices_outliers_iqr = set()
        indices_outliers_kmeans = set()
        indices_outliers_svm = set()
        
        # MÉTODO IQR
        if metodo in ['iqr', 'combinado']:
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers_iqr_mask = (data < lower_bound) | (data > upper_bound)
            outliers_iqr = data[outliers_iqr_mask]
            n_outliers_iqr = len(outliers_iqr)
            indices_outliers_iqr = set(data[outliers_iqr_mask].index)
            
            if metodo == 'iqr':
                n_outliers = n_outliers_iqr
                if n_outliers > 0:
                    outlier_info.update({
                        'cantidad': n_outliers,
                        'porcentaje': (n_outliers / len(data)) * 100,
                        'limite_inferior': lower_bound,
                        'limite_superior': upper_bound,
                        'min_outlier': outliers_iqr.min(),
                        'max_outlier': outliers_iqr.max()
                    })
        
        # MÉTODO K-MEANS
        if metodo in ['kmeans', 'combinado'] and len(data) >= 3:
            try:
                data_reshaped = data.values.reshape(-1, 1)
                scaler = StandardScaler()
                data_scaled = scaler.fit_transform(data_reshaped)
                
                kmeans = KMeans(n_clusters=min(3, len(data)), random_state=42, n_init=10)
                kmeans.fit(data_scaled)
                distances = np.min(kmeans.transform(data_scaled), axis=1)
                threshold = np.percentile(distances, 90)
                
                outliers_kmeans_mask = distances > threshold
                n_outliers_kmeans = outliers_kmeans_mask.sum()
                indices_outliers_kmeans = set(data.index[outliers_kmeans_mask])
                
                if metodo == 'kmeans':
                    n_outliers = n_outliers_kmeans
                    if n_outliers > 0:
                        outliers_kmeans_data = data[outliers_kmeans_mask]
                        outlier_info.update({
                            'cantidad': n_outliers,
                            'porcentaje': (n_outliers / len(data)) * 100,
                            'threshold': threshold,
                            'min_outlier': outliers_kmeans_data.min(),
                            'max_outlier': outliers_kmeans_data.max()
                        })
            except:
                n_outliers_kmeans = 0
                indices_outliers_kmeans = set()
        
        # MÉTODO SVM
        if metodo in ['svm', 'combinado'] and len(data) >= 10:
            try:
                data_reshaped = data.values.reshape(-1, 1)
                scaler = StandardScaler()
                data_scaled = scaler.fit_transform(data_reshaped)
                
                svm = OneClassSVM(nu=0.1, kernel='rbf', gamma='auto')
                predictions = svm.fit_predict(data_scaled)
                
                outliers_svm_mask = predictions == -1
                n_outliers_svm = outliers_svm_mask.sum()
                indices_outliers_svm = set(data.index[outliers_svm_mask])
                
                if metodo == 'svm':
                    n_outliers = n_outliers_svm
                    if n_outliers > 0:
                        outliers_svm_data = data[outliers_svm_mask]
                        outlier_info.update({
                            'cantidad': n_outliers,
                            'porcentaje': (n_outliers / len(data)) * 100,
                            'min_outlier': outliers_svm_data.min(),
                            'max_outlier': outliers_svm_data.max()
                        })
            except:
                n_outliers_svm = 0
                indices_outliers_svm = set()
        
        # MÉTODO COMBINADO (unión de índices únicos)
        if metodo == 'combinado':
            # Unión de índices únicos para evitar conteos duplicados
            indices_unicos = indices_outliers_iqr | indices_outliers_kmeans | indices_outliers_svm
            n_outliers = len(indices_unicos)
            
            if n_outliers > 0:
                outlier_info.update({
                    'cantidad': n_outliers,
                    'porcentaje': (n_outliers / len(data)) * 100,
                    'outliers_iqr': n_outliers_iqr,
                    'outliers_kmeans': n_outliers_kmeans,
                    'outliers_svm': n_outliers_svm,
                    'outliers_unicos': n_outliers,
                    'overlapping': n_outliers_iqr + n_outliers_kmeans + n_outliers_svm - n_outliers,
                    'indices_outliers': list(indices_unicos)
                })
        
        # Guardar índices según el método usado
        if metodo == 'iqr':
            indices_metodo = indices_outliers_iqr
        elif metodo == 'kmeans':
            indices_metodo = indices_outliers_kmeans
        elif metodo == 'svm':
            indices_metodo = indices_outliers_svm
        else:  # combinado
            indices_metodo = indices_outliers_iqr | indices_outliers_kmeans | indices_outliers_svm
        
        # Agregar índices de esta variable al conjunto global
        todos_indices_outliers.update(indices_metodo)
        
        if n_outliers > 0:
            if 'indices_outliers' not in outlier_info:
                outlier_info['indices_outliers'] = list(indices_metodo)
            outliers_por_columna[col] = outlier_info
        
        total_outliers += n_outliers
        total_datos += len(data)
    
    if total_datos > 0:
        pct_datos_precisos = ((total_datos - total_outliers) / total_datos) * 100
        if pct_datos_precisos >= 95:
            score = 20
        elif pct_datos_precisos >= 85:
            score = 15 + ((pct_datos_precisos - 85) / 10) * 5
        elif pct_datos_precisos >= 70:
            score = 5 + ((pct_datos_precisos - 70) / 15) * 10
        else:
            score = (pct_datos_precisos / 70) * 5
    else:
        pct_datos_precisos = 100
        score = 20
    
    # Crear DataFrame con información completa de las filas con outliers
    if len(todos_indices_outliers) > 0:
        df_outliers_completo = df.loc[list(todos_indices_outliers)].copy()
        # Agregar columna indicando en qué variables tiene outliers
        variables_outlier = []
        for idx in df_outliers_completo.index:
            vars_con_outlier = []
            for col, info in outliers_por_columna.items():
                if idx in info.get('indices_outliers', []):
                    vars_con_outlier.append(col)
            variables_outlier.append(', '.join(vars_con_outlier) if vars_con_outlier else '')
        df_outliers_completo['variables_con_outlier'] = variables_outlier
    else:
        df_outliers_completo = pd.DataFrame()
    
    return {
        'score': score,
        'pct_datos_precisos': pct_datos_precisos,
        'outliers_por_columna': outliers_por_columna,
        'total_outliers': total_outliers,
        'total_datos_numericos': total_datos,
        'metodo_usado': metodo,
        'df_outliers_completo': df_outliers_completo,
        'num_filas_con_outliers': len(todos_indices_outliers)
    }

def calcular_variabilidad(df: pd.DataFrame, variables_numericas: List[str] = None) -> Dict:
    """Calcula la variabilidad controlada (CV razonable)"""
    # Usar DataFrame convertido a numérico
    df_numeric = preparar_dataframe_numerico(df, variables_numericas)
    
    if df_numeric.empty:
        return {
            'score': 15,
            'cv_promedio': 0,
            'cv_por_columna': {},
            'columnas_variabilidad_extrema': {},
            'pct_variabilidad_adecuada': 100
        }
    
    cv_por_columna = {}
    columnas_variabilidad_extrema = {}
    cvs_validos = []
    
    for col in df_numeric.columns:
        data = df_numeric[col].dropna()
        if len(data) > 0 and data.mean() != 0:
            cv = (data.std() / data.mean()) * 100
            cv_por_columna[col] = cv
            cvs_validos.append(abs(cv))
            
            if abs(cv) > 200:
                columnas_variabilidad_extrema[col] = {
                    'cv': cv,
                    'problema': 'Variabilidad excesiva'
                }
            elif abs(cv) < 1 and data.nunique() > 1:
                columnas_variabilidad_extrema[col] = {
                    'cv': cv,
                    'problema': 'Variabilidad muy baja'
                }
    
    n_columnas = len(cv_por_columna)
    n_extremas = len(columnas_variabilidad_extrema)
    
    if n_columnas > 0:
        pct_adecuadas = ((n_columnas - n_extremas) / n_columnas) * 100
        score = (pct_adecuadas / 100) * 15
    else:
        pct_adecuadas = 100
        score = 15
    
    cv_promedio = np.mean(cvs_validos) if cvs_validos else 0
    
    return {
        'score': score,
        'cv_promedio': cv_promedio,
        'cv_por_columna': cv_por_columna,
        'columnas_variabilidad_extrema': columnas_variabilidad_extrema,
        'pct_variabilidad_adecuada': pct_adecuadas
    }

def calcular_integridad(df: pd.DataFrame, columnas_esperadas: List[str] = None) -> Dict:
    """Calcula la integridad estructural de los datos"""
    if columnas_esperadas is None or len(columnas_esperadas) == 0:
        score = 10
        columnas_faltantes = []
        columnas_extra = []
        pct_integridad = 100
    else:
        columnas_actuales = set(df.columns)
        columnas_esperadas_set = set(columnas_esperadas)
        
        columnas_faltantes = list(columnas_esperadas_set - columnas_actuales)
        columnas_extra = list(columnas_actuales - columnas_esperadas_set)
        
        columnas_coincidentes = len(columnas_esperadas_set & columnas_actuales)
        pct_integridad = (columnas_coincidentes / len(columnas_esperadas_set)) * 100
        
        score = (pct_integridad / 100) * 10
    
    return {
        'score': score,
        'pct_integridad': pct_integridad,
        'columnas_faltantes': columnas_faltantes,
        'columnas_extra': columnas_extra,
        'total_columnas': len(df.columns)
    }

def calcular_indice_calidad_datos(
    df: pd.DataFrame, 
    variables_numericas: List[str] = None,
    columnas_esperadas: List[str] = None,
    metodo_outliers: str = 'iqr'
) -> Dict:
    """Calcula el Índice de Calidad de Datos completo (0-100)"""
    # Pasar variables_numericas a todas las funciones para análisis consistente
    completitud = calcular_completitud(df, variables_numericas)
    unicidad = calcular_unicidad(df, variables_numericas)
    consistencia = calcular_consistencia(df, variables_numericas)
    precision = calcular_precision_outliers(df, variables_numericas, metodo=metodo_outliers)
    variabilidad = calcular_variabilidad(df, variables_numericas)
    integridad = calcular_integridad(df, columnas_esperadas)
    
    icd_total = (
        completitud['score'] +
        unicidad['score'] +
        consistencia['score'] +
        precision['score'] +
        variabilidad['score'] +
        integridad['score']
    )
    
    if icd_total >= 90:
        nivel_calidad = "Excelente"
        color = "green"
        emoji = "🟢"
    elif icd_total >= 75:
        nivel_calidad = "Buena"
        color = "lightgreen"
        emoji = "🟡"
    elif icd_total >= 60:
        nivel_calidad = "Aceptable"
        color = "orange"
        emoji = "🟠"
    elif icd_total >= 40:
        nivel_calidad = "Baja"
        color = "orangered"
        emoji = "🟠"
    else:
        nivel_calidad = "Crítica"
        color = "red"
        emoji = "🔴"
    
    return {
        'icd_total': round(icd_total, 2),
        'nivel_calidad': nivel_calidad,
        'color': color,
        'emoji': emoji,
        'desglose': {
            'Completitud (25pts)': round(completitud['score'], 2),
            'Unicidad (15pts)': round(unicidad['score'], 2),
            'Consistencia (15pts)': round(consistencia['score'], 2),
            'Precisión (20pts)': round(precision['score'], 2),
            'Variabilidad (15pts)': round(variabilidad['score'], 2),
            'Integridad (10pts)': round(integridad['score'], 2)
        },
        'detalles': {
            'completitud': completitud,
            'unicidad': unicidad,
            'consistencia': consistencia,
            'precision': precision,
            'variabilidad': variabilidad,
            'integridad': integridad
        }
    }

def generar_recomendaciones(resultado_icd: Dict) -> List[str]:
    """Genera recomendaciones basadas en el ICD calculado"""
    recomendaciones = []
    detalles = resultado_icd['detalles']
    
    if detalles['completitud']['score'] < 20:
        pct = detalles['completitud']['pct_completo']
        recomendaciones.append(
            f"⚠️ **Completitud baja ({pct:.1f}%)**: Considerar imputación de valores faltantes o eliminar columnas/filas muy incompletas."
        )
    
    if detalles['unicidad']['filas_duplicadas'] > 0:
        n_dup = detalles['unicidad']['filas_duplicadas']
        recomendaciones.append(
            f"⚠️ **{n_dup} filas duplicadas detectadas**: Revisar si son registros legítimos o errores de carga."
        )
    
    if detalles['precision']['score'] < 15:
        pct = detalles['precision']['pct_datos_precisos']
        n_out = detalles['precision']['total_outliers']
        recomendaciones.append(
            f"⚠️ **Outliers significativos ({n_out} detectados, {100-pct:.1f}% de datos)**: Revisar valores atípicos y validar si son errores o valores legítimos."
        )
    
    if len(detalles['variabilidad']['columnas_variabilidad_extrema']) > 0:
        n_extremas = len(detalles['variabilidad']['columnas_variabilidad_extrema'])
        recomendaciones.append(
            f"⚠️ **{n_extremas} columnas con variabilidad extrema**: Revisar escalas de medición o transformar datos."
        )
    
    if detalles['consistencia']['valores_inconsistentes'] > 0:
        n_incons = detalles['consistencia']['valores_inconsistentes']
        recomendaciones.append(
            f"⚠️ **{n_incons} valores inconsistentes**: Estandarizar formatos y tipos de datos."
        )
    
    if not recomendaciones:
        recomendaciones.append("✅ **Los datos tienen buena calidad general**. Listo para análisis.")
    
    return recomendaciones

# ============================================================================
# CLASES
# ============================================================================

class DataCleaner:
    """Limpiador automático de datos"""
    
    @staticmethod
    def clean_dataframe(df: pd.DataFrame, show_report: bool = True) -> tuple:
        """Limpia el DataFrame eliminando filas/columnas vacías, duplicados, etc."""
        original_shape = df.shape
        cleaning_report = []
        
        rows_before = len(df)
        df = df.dropna(how='all')
        rows_removed = rows_before - len(df)
        if rows_removed > 0:
            cleaning_report.append(f"🗑️ Eliminadas {rows_removed} filas completamente vacías")
        
        cols_before = len(df.columns)
        df = df.dropna(axis=1, how='all')
        cols_removed = cols_before - len(df.columns)
        if cols_removed > 0:
            cleaning_report.append(f"🗑️ Eliminadas {cols_removed} columnas completamente vacías")
        
        numeric_cols_cleaned = 0
        for col in df.columns:
            if df[col].dtype == 'object':
                sample = df[col].dropna().head(100)
                if len(sample) > 0:
                    numeric_count = 0
                    for val in sample:
                        try:
                            float(str(val).replace(',', '.').strip())
                            numeric_count += 1
                        except:
                            pass
                    
                    if numeric_count / len(sample) > 0.8:
                        original_nulls = df[col].isna().sum()
                        df[col] = pd.to_numeric(
                            df[col].astype(str).str.replace(',', '.').str.strip(),
                            errors='coerce'
                        )
                        new_nulls = df[col].isna().sum()
                        invalid_values = new_nulls - original_nulls
                        
                        if invalid_values > 0:
                            cleaning_report.append(
                                f"🔢 Columna '{col}': convertida a numérica ({invalid_values} valores inválidos → NaN)"
                            )
                            numeric_cols_cleaned += 1
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            rows_before = len(df)
            df = df.dropna(subset=numeric_columns, how='all')
            rows_removed = rows_before - len(df)
            if rows_removed > 0:
                cleaning_report.append(
                    f"🗑️ Eliminadas {rows_removed} filas sin datos numéricos válidos"
                )
        
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            df = df.drop_duplicates()
            cleaning_report.append(f"🗑️ Eliminadas {duplicates} filas duplicadas")
        
        text_cols_cleaned = 0
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace(['nan', 'None', 'NaN', ''], np.nan)
            text_cols_cleaned += 1
        
        if text_cols_cleaned > 0:
            cleaning_report.append(f"✨ Limpiados espacios en {text_cols_cleaned} columnas de texto")
        
        df = df.reset_index(drop=True)
        
        final_shape = df.shape
        cleaning_report.insert(0, f"📊 Dimensiones: {original_shape} → {final_shape}")
        
        return df, cleaning_report
    
    @staticmethod
    def show_cleaning_report(report: list):
        """Muestra el reporte de limpieza"""
        if report:
            st.success("✅ Limpieza completada")
            with st.expander("📋 Ver reporte de limpieza", expanded=True):
                for item in report:
                    st.write(item)

# ============================================================================
# FUNCIONES DE CARGA DE DATOS
# ============================================================================

def load_data_from_socrata(domain: str, dataset_id: str, limit: int, app_token: str = None) -> tuple:
    """Carga datos desde Socrata API"""
    try:
        client = Socrata(
            domain,
            app_token,
            timeout=30
        )
        
        results = client.get(dataset_id, limit=limit)
        df = pd.DataFrame.from_records(results)
        
        return df, None
        
    except Exception as e:
        return None, str(e)

# ============================================================================
# FUNCIONES DE ESTADÍSTICOS Y VISUALIZACIONES
# ============================================================================

def calcular_estadisticos(df: pd.DataFrame, variables: list) -> pd.DataFrame:
    """Calcula estadísticos descriptivos para las variables especificadas"""
    # Usar función centralizada de conversión
    df_numeric = preparar_dataframe_numerico(df, variables)
    
    if df_numeric.empty:
        return None
    
    # Calcular estadísticos con manejo de división por cero en CV
    means = df_numeric.mean()
    stds = df_numeric.std()
    
    # Calcular CV evitando división por cero
    cv_values = []
    for col in df_numeric.columns:
        if means[col] != 0:
            cv_values.append((stds[col] / means[col]) * 100)
        else:
            cv_values.append(np.nan)
    
    # DETECCIÓN DE OUTLIERS POR VARIABLE
    outliers_iqr = []
    outliers_kmeans = []
    outliers_svm = []
    outliers_total = []
    
    for col in df_numeric.columns:
        data = df_numeric[col].dropna()
        n_outliers_iqr = 0
        n_outliers_kmeans = 0
        n_outliers_svm = 0
        
        if len(data) > 0:
            # 1. MÉTODO IQR
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            n_outliers_iqr = ((data < lower_bound) | (data > upper_bound)).sum()
            
            # 2. MÉTODO K-MEANS (univariado - basado en distancia al centroide)
            if len(data) >= 3:
                try:
                    data_reshaped = data.values.reshape(-1, 1)
                    scaler = StandardScaler()
                    data_scaled = scaler.fit_transform(data_reshaped)
                    
                    kmeans = KMeans(n_clusters=min(3, len(data)), random_state=42, n_init=10)
                    kmeans.fit(data_scaled)
                    distances = np.min(kmeans.transform(data_scaled), axis=1)
                    threshold = np.percentile(distances, 90)  # 10% como outliers
                    n_outliers_kmeans = (distances > threshold).sum()
                except:
                    n_outliers_kmeans = 0
            
            # 3. MÉTODO SVM (One-Class SVM univariado)
            if len(data) >= 10:
                try:
                    data_reshaped = data.values.reshape(-1, 1)
                    scaler = StandardScaler()
                    data_scaled = scaler.fit_transform(data_reshaped)
                    
                    svm = OneClassSVM(nu=0.1, kernel='rbf', gamma='auto')
                    predictions = svm.fit_predict(data_scaled)
                    n_outliers_svm = (predictions == -1).sum()
                except:
                    n_outliers_svm = 0
        
        outliers_iqr.append(n_outliers_iqr)
        outliers_kmeans.append(n_outliers_kmeans)
        outliers_svm.append(n_outliers_svm)
        outliers_total.append(n_outliers_iqr + n_outliers_kmeans + n_outliers_svm)
    
    stats = pd.DataFrame({
        'Variable': df_numeric.columns,
        'Count': df_numeric.count().values,
        'Media': means.values,
        'Mediana': df_numeric.median().values,
        'Desv. Std': stds.values,
        'Mínimo': df_numeric.min().values,
        'Q1 (25%)': df_numeric.quantile(0.25).values,
        'Q3 (75%)': df_numeric.quantile(0.75).values,
        'Máximo': df_numeric.max().values,
        'Rango': (df_numeric.max() - df_numeric.min()).values,
        'CV (%)': cv_values,
        'Asimetría': df_numeric.skew().values,
        'Curtosis': df_numeric.kurtosis().values,
        'Valores nulos': df_numeric.isnull().sum().values,
        'Outliers IQR': outliers_iqr,
        'Outliers K-means': outliers_kmeans,
        'Outliers SVM': outliers_svm,
        'Total Outliers': outliers_total
    })
    
    return stats

def crear_histogramas(df: pd.DataFrame, variables: list):
    """Crea histogramas para las variables seleccionadas"""
    df_numeric = preparar_dataframe_numerico(df, variables)
    
    if df_numeric.empty:
        return None
    
    n_vars = len(df_numeric.columns)
    n_cols = 3
    n_rows = (n_vars + n_cols - 1) // n_cols
    
    fig = make_subplots(
        rows=n_rows, 
        cols=n_cols,
        subplot_titles=df_numeric.columns.tolist(),
        vertical_spacing=0.12,
        horizontal_spacing=0.08
    )
    
    for idx, col in enumerate(df_numeric.columns):
        row = idx // n_cols + 1
        col_pos = idx % n_cols + 1
        
        data = df_numeric[col].dropna()
        
        fig.add_trace(
            go.Histogram(
                x=data,
                name=col,
                marker_color='steelblue',
                showlegend=False
            ),
            row=row,
            col=col_pos
        )
    
    fig.update_layout(
        height=300 * n_rows,
        title_text="Distribución de Variables",
        showlegend=False
    )
    
    return fig

def crear_boxplots(df: pd.DataFrame, variables: list):
    """Crea boxplots para las variables seleccionadas"""
    df_numeric = preparar_dataframe_numerico(df, variables)
    
    if df_numeric.empty:
        return None
    
    n_vars = len(df_numeric.columns)
    n_cols = 3
    n_rows = (n_vars + n_cols - 1) // n_cols
    
    fig = make_subplots(
        rows=n_rows, 
        cols=n_cols,
        subplot_titles=df_numeric.columns.tolist(),
        vertical_spacing=0.12,
        horizontal_spacing=0.08
    )
    
    for idx, col in enumerate(df_numeric.columns):
        row = idx // n_cols + 1
        col_pos = idx % n_cols + 1
        
        data = df_numeric[col].dropna()
        
        fig.add_trace(
            go.Box(
                y=data,
                name=col,
                marker_color='lightseagreen',
                showlegend=False
            ),
            row=row,
            col=col_pos
        )
    
    fig.update_layout(
        height=300 * n_rows,
        title_text="Boxplots de Variables",
        showlegend=False
    )
    
    return fig

def crear_matriz_correlacion(df: pd.DataFrame, variables: list):
    """Crea matriz de correlación para las variables seleccionadas"""
    df_numeric = preparar_dataframe_numerico(df, variables)
    
    if df_numeric.empty or len(df_numeric.columns) < 2:
        return None
    
    corr_matrix = df_numeric.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 8},
        colorbar=dict(title="Correlacion")
    ))
    
    fig.update_layout(
        title="Matriz de Correlacion",
        xaxis_title="Variables",
        yaxis_title="Variables",
        height=600,
        width=800
    )
    
    return fig

# ============================================================================
# INICIALIZACIÓN DE SESSION STATE
# ============================================================================

if 'df' not in st.session_state:
    st.session_state.df = None
if 'df_original' not in st.session_state:
    st.session_state.df_original = None
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'agent_config_key' not in st.session_state:
    st.session_state.agent_config_key = None
if 'data_source' not in st.session_state:
    st.session_state.data_source = None
if 'cleaning_report' not in st.session_state:
    st.session_state.cleaning_report = None
if 'select_all_vars' not in st.session_state:
    st.session_state.select_all_vars = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# ============================================================================
# INTERFAZ PRINCIPAL
# ============================================================================

st.title("📊 Agente datos suelos Agrosavia")
st.markdown("**Carga tu archivo CSV/XLS o usa la API Socrata y haz preguntas sobre tus datos usando IA**")

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
    
    st.divider()
    
    if st.session_state.df is not None:
        st.subheader("📋 Dataset")
        st.info(f"📏 {st.session_state.df.shape[0]} filas × {st.session_state.df.shape[1]} columnas")
        
        if st.session_state.data_source:
            st.caption(f"🔗 Fuente: {st.session_state.data_source}")
        
        with st.expander("ℹ️ Tipos de datos"):
            numeric_cols = st.session_state.df.select_dtypes(include=[np.number]).columns
            text_cols = st.session_state.df.select_dtypes(include=['object']).columns
            
            st.write(f"**Numéricas:** {len(numeric_cols)}")
            st.write(f"**Texto:** {len(text_cols)}")
            
            null_total = st.session_state.df.isnull().sum().sum()
            st.write(f"**Valores nulos:** {null_total}")
        
        if st.button("🗑️ Limpiar todo", use_container_width=True):
            st.session_state.df = None
            st.session_state.df_original = None
            st.session_state.agent = None
            st.session_state.agent_config_key = None
            st.session_state.data_source = None
            st.session_state.cleaning_report = None
            st.session_state.chat_history = []
            st.rerun()

# Verificar si se ha ingresado la API key
if not openai_api_key:
    st.warning("⚠️ Por favor, ingresa tu API Key de OpenAI en la barra lateral.")
    st.info("Puedes obtener tu API key en: https://platform.openai.com/api-keys")

# Configurar la variable de entorno
os.environ["OPENAI_API_KEY"] = openai_api_key

# Tabs principales
tab1, tab2, tab3 = st.tabs(["📁 Cargar Datos", "📊 Estadísticos", "🤖 Análisis con IA"])

# TAB 1: CARGAR DATOS
with tab1:
    subtab1, subtab2 = st.tabs(["📁 Archivo CSV/Excel", "🌐 API Socrata"])
    
    with subtab1:
        uploaded_file = st.file_uploader(
            "Selecciona un archivo CSV o Excel:",
            type=['csv', 'xlsx', 'xls'],
            help="Formatos soportados: CSV, Excel (.xlsx, .xls)"
        )
        
        if uploaded_file is not None:
            try:
                # Leer el archivo según su tipo
                if uploaded_file.name.endswith('.csv'):
                    df_raw = pd.read_csv(uploaded_file, on_bad_lines='skip')
                else:
                    df_raw = pd.read_excel(uploaded_file)
                
                # Asignar tipos de datos correctos
                df_raw = asignar_tipos_datos(df_raw)
                
                st.session_state.df_original = df_raw.copy()
                
                if st.checkbox("🧹 Aplicar limpieza automática", value=False, key="clean_file"):
                    with st.spinner("🧹 Limpiando datos..."):
                        df_clean, report = DataCleaner.clean_dataframe(df_raw)
                        st.session_state.df = df_clean
                        st.session_state.cleaning_report = report
                        st.session_state.data_source = f"Archivo: {uploaded_file.name}"
                    
                    DataCleaner.show_cleaning_report(report)
                else:
                    st.session_state.df = df_raw.copy()
                    st.session_state.data_source = f"Archivo: {uploaded_file.name} (sin limpiar)"
                    st.info("📊 Usando datos originales (sin limpieza)")
                
                st.success(f"✅ Archivo cargado exitosamente: {uploaded_file.name}")
                
                # Mostrar información básica del dataset
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📏 Filas", st.session_state.df.shape[0])
                with col2:
                    st.metric("📊 Columnas", st.session_state.df.shape[1])
                with col3:
                    st.metric("💾 Tamaño", f"{st.session_state.df.memory_usage(deep=True).sum() / 1024:.1f} KB")
                
                with st.expander("👀 Vista previa de datos", expanded=False):
                    st.dataframe(st.session_state.df.head(10), use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Error al cargar el archivo: {str(e)}")
                st.info("Verifica que el archivo tenga el formato correcto (CSV o Excel).")
    
    with subtab2:
        st.markdown("### 🌐 Configuración de Socrata")
        
        col1, col2 = st.columns(2)
        
        with col1:
            domain = st.text_input(
                "🌍 Dominio:",
                value="www.datos.gov.co",
                help="Ejemplo: www.datos.gov.co"
            )
        
        with col2:
            dataset_id = st.text_input(
                "🆔 Dataset ID:",
                value="ch4u-f3i5",
                help="Ejemplo: ch4u-f3i5"
            )
        
        col3, col4 = st.columns(2)
        
        with col3:
            limit = st.number_input(
                "📊 Límite de registros:",
                min_value=100,
                max_value=50000,
                value=2000,
                step=500
            )
        
        with col4:
            app_token = st.text_input(
                "🔑 App Token (opcional):",
                type="password",
                help="Token de aplicación de Socrata"
            )
        
        if st.button("🔄 Cargar desde API", use_container_width=True, type="primary"):
            if domain and dataset_id:
                with st.spinner("📥 Cargando datos desde API..."):
                    df_raw, error = load_data_from_socrata(
                        domain=domain,
                        dataset_id=dataset_id,
                        limit=limit,
                        app_token=app_token if app_token else None
                    )
                    
                    if df_raw is not None:
                        # Asignar tipos de datos correctos
                        df_raw = asignar_tipos_datos(df_raw)
                        
                        st.session_state.df_original = df_raw.copy()
                        st.session_state.df = df_raw.copy()
                        st.session_state.data_source = f"API: {domain}/{dataset_id}"
                        
                        st.success(f"✅ Datos cargados exitosamente desde API")
                        
                        # Mostrar información básica del dataset
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("📏 Filas", st.session_state.df.shape[0])
                        with col2:
                            st.metric("📊 Columnas", st.session_state.df.shape[1])
                        with col3:
                            st.metric("💾 Tamaño", f"{st.session_state.df.memory_usage(deep=True).sum() / 1024:.1f} KB")
                        
                        with st.expander("👀 Vista previa de datos", expanded=True):
                            st.dataframe(st.session_state.df.head(10), use_container_width=True)
                    else:
                        st.error(f"❌ Error al cargar datos: {error}")
            else:
                st.warning("⚠️ Por favor completa los campos requeridos")
    
    if st.session_state.df is not None and st.session_state.df_original is not None:
        with st.expander("🔍 Comparar datos originales vs procesados"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📋 Datos Originales")
                st.write(f"Shape: {st.session_state.df_original.shape}")
                st.dataframe(st.session_state.df_original.head(5), use_container_width=True)
            
            with col2:
                st.subheader("✨ Datos Procesados")
                st.write(f"Shape: {st.session_state.df.shape}")
                st.dataframe(st.session_state.df.head(5), use_container_width=True)

# TAB 2: ESTADÍSTICOS
with tab2:
    if st.session_state.df_original is not None:
        st.header("📊 Análisis Estadístico de Variables")
        
        vars_disponibles = [v for v in VARIABLES_ESTADISTICAS if v in st.session_state.df_original.columns]
        vars_no_disponibles = [v for v in VARIABLES_ESTADISTICAS if v not in st.session_state.df_original.columns]
        
        if vars_disponibles:
            st.success(f"✅ Se encontraron {len(vars_disponibles)} de {len(VARIABLES_ESTADISTICAS)} variables")
            
            if vars_no_disponibles:
                with st.expander("⚠️ Variables no encontradas en el dataset"):
                    for var in vars_no_disponibles:
                        st.write(f"- {var}")
            
            st.subheader("🔧 Configuración de análisis")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                # Determinar el valor default basado en el estado
                default_vars = vars_disponibles if st.session_state.select_all_vars else []
                
                variables_seleccionadas = st.multiselect(
                    "Selecciona variables para analizar:",
                    options=vars_disponibles,
                    default=default_vars,
                    help="Selecciona las variables que deseas incluir en el análisis"
                )
                
                # Selector de método para detección de outliers en ICD
                metodo_outliers = st.selectbox(
                    "🎯 Método de detección de outliers para ICD:",
                    options=['iqr', 'kmeans', 'svm', 'combinado'],
                    format_func=lambda x: {
                        'iqr': '📊 IQR (Cuartiles) - Tradicional',
                        'kmeans': '🎯 K-means - Clustering',
                        'svm': '🤖 SVM - One-Class',
                        'combinado': '🔄 Combinado (suma de los 3)'
                    }[x],
                    help="Selecciona el método para calcular la dimensión de Precisión en el ICD"
                )
            
            with col2:
                st.write("")
                st.write("")
                # Botón para seleccionar todas
                if st.button("✅ Seleccionar Todas", use_container_width=True):
                    st.session_state.select_all_vars = True
                    st.rerun()
                analizar_btn = st.button("📈 Generar Análisis", type="primary", use_container_width=True)
            
            # Resetear el flag si el usuario cambia la selección manualmente
            if not st.session_state.select_all_vars and len(variables_seleccionadas) == len(vars_disponibles):
                st.session_state.select_all_vars = True
            elif st.session_state.select_all_vars and len(variables_seleccionadas) != len(vars_disponibles):
                st.session_state.select_all_vars = False
            
            if analizar_btn and variables_seleccionadas:
                with st.spinner("📊 Generando análisis..."):
                    # Usar df_original para ICD (sin limpiar) y df para estadísticos (limpio o procesado)
                    stats_df = calcular_estadisticos(st.session_state.df_original, variables_seleccionadas)
                    
                    if stats_df is not None:
                        st.divider()
                        
                        # SECCIÓN 1: Estadísticos descriptivos
                        st.subheader("📋 Estadísticos Descriptivos")
                        
                        stats_display = stats_df.copy()
                        numeric_columns = ['Media', 'Mediana', 'Desv. Std', 'Mínimo', 'Q1 (25%)', 'Q3 (75%)', 'Máximo', 'Rango', 'CV (%)', 'Asimetría', 'Curtosis']
                        for col in numeric_columns:
                            if col in stats_display.columns:
                                stats_display[col] = stats_display[col].round(3)
                        
                        # Resaltar columnas de outliers
                        st.info("💡 **Detección de Outliers por 3 métodos:** IQR (Cuartiles), K-means (Clustering), SVM (One-Class)")
                        
                        st.dataframe(
                            stats_display,
                            use_container_width=True,
                            hide_index=True,
                            height=400
                        )
                        
                        # Análisis de outliers
                        if 'Total Outliers' in stats_display.columns:
                            total_outliers_sum = stats_display['Total Outliers'].sum()
                            if total_outliers_sum > 0:
                                st.warning(f"⚠️ **Total de outliers detectados (suma de 3 métodos): {int(total_outliers_sum)}**")
                                
                                # Mostrar variables con más outliers
                                top_outliers = stats_display.nlargest(5, 'Total Outliers')[['Variable', 'Outliers IQR', 'Outliers K-means', 'Outliers SVM', 'Total Outliers']]
                                if len(top_outliers) > 0:
                                    st.markdown("**🔍 Variables con más outliers detectados:**")
                                    st.dataframe(top_outliers, use_container_width=True, hide_index=True)
                            else:
                                st.success("✅ No se detectaron outliers significativos en las variables analizadas")
                        
                        csv = stats_display.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Descargar estadísticos como CSV",
                            data=csv,
                            file_name="estadisticos_variables.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                        
                        st.divider()
                        
                        # SECCIÓN 2: ÍNDICE DE CALIDAD DE DATOS
                        st.subheader("🎯 Índice de Calidad de Datos (ICD)")
                        
                        with st.spinner("Calculando índice de calidad..."):
                            resultado_icd = calcular_indice_calidad_datos(
                                df=st.session_state.df_original,
                                variables_numericas=variables_seleccionadas,
                                columnas_esperadas=VARIABLES_ESTADISTICAS,
                                metodo_outliers=metodo_outliers
                            )
                        
                        # Métrica principal
                        st.markdown("### 📊 Calidad General")
                        col_metric1, col_metric2, col_metric3 = st.columns([2, 1, 1])
                        
                        with col_metric1:
                            icd_total = resultado_icd['icd_total']
                            nivel = resultado_icd['nivel_calidad']
                            emoji = resultado_icd['emoji']
                            
                            st.markdown(f"""
                            <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                        border-radius: 10px; color: white;'>
                                <h1 style='margin: 0; font-size: 60px;'>{emoji} {icd_total:.1f}</h1>
                                <h3 style='margin: 10px 0 0 0;'>Calidad {nivel}</h3>
                                <p style='margin: 5px 0 0 0; opacity: 0.9;'>sobre 100 puntos</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col_metric2:
                            st.metric(
                                "Variables analizadas",
                                len(variables_seleccionadas),
                                help="Número de variables incluidas"
                            )
                            
                            completitud_pct = resultado_icd['detalles']['completitud']['pct_completo']
                            st.metric(
                                "Completitud",
                                f"{completitud_pct:.1f}%",
                                help="% de datos no nulos"
                            )
                        
                        with col_metric3:
                            unicidad_pct = resultado_icd['detalles']['unicidad']['pct_registros_unicos']
                            st.metric(
                                "Unicidad",
                                f"{unicidad_pct:.1f}%",
                                help="% de registros únicos"
                            )
                            
                            precision_pct = resultado_icd['detalles']['precision']['pct_datos_precisos']
                            st.metric(
                                "Precisión",
                                f"{precision_pct:.1f}%",
                                help="% de datos sin outliers"
                            )
                        
                        st.markdown("---")
                        
                        # Desglose por dimensiones
                        st.markdown("### 📈 Desglose por Dimensiones")
                        
                        col1, col2, col3 = st.columns(3)
                        desglose = resultado_icd['desglose']
                        
                        with col1:
                            st.metric(
                                "🔵 Completitud",
                                f"{desglose['Completitud (25pts)']:.1f} / 25",
                                help="Valores no nulos y columnas completas"
                            )
                            st.metric(
                                "🟣 Unicidad",
                                f"{desglose['Unicidad (15pts)']:.1f} / 15",
                                help="Ausencia de duplicados"
                            )
                        
                        with col2:
                            st.metric(
                                "🟢 Consistencia",
                                f"{desglose['Consistencia (15pts)']:.1f} / 15",
                                help="Coherencia en tipos de datos"
                            )
                            st.metric(
                                "🟡 Precisión",
                                f"{desglose['Precisión (20pts)']:.1f} / 20",
                                help="Datos sin outliers extremos"
                            )
                        
                        with col3:
                            st.metric(
                                "🟠 Variabilidad",
                                f"{desglose['Variabilidad (15pts)']:.1f} / 15",
                                help="CV en rangos razonables"
                            )
                            st.metric(
                                "🔴 Integridad",
                                f"{desglose['Integridad (10pts)']:.1f} / 10",
                                help="Estructura completa"
                            )
                        
                        st.markdown("---")
                        
                        # Métricas detalladas
                        st.markdown("### 🔍 Métricas Detalladas")
                        
                        tab_comp, tab_uni, tab_prec, tab_var = st.tabs([
                            "📊 Completitud", 
                            "🔄 Unicidad", 
                            "🎯 Precisión (Outliers)", 
                            "📉 Variabilidad"
                        ])
                        
                        with tab_comp:
                            detalles_comp = resultado_icd['detalles']['completitud']
                            
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Valores totales", f"{detalles_comp['total_valores']:,}")
                            col2.metric("Valores nulos", f"{detalles_comp['total_nulos']:,}")
                            col3.metric("Completitud", f"{detalles_comp['pct_completo']:.1f}%")
                            
                            if detalles_comp['columnas_problematicas']:
                                st.warning("⚠️ **Columnas con >50% de valores nulos:**")
                                df_prob = pd.DataFrame([
                                    {'Columna': col, '% Nulos': f"{pct:.1f}%"} 
                                    for col, pct in detalles_comp['columnas_problematicas'].items()
                                ])
                                st.dataframe(df_prob, use_container_width=True, hide_index=True)
                            else:
                                st.success("✅ Todas las columnas tienen menos del 50% de valores nulos")
                        
                        with tab_uni:
                            detalles_uni = resultado_icd['detalles']['unicidad']
                            
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Total filas", f"{len(st.session_state.df_original):,}")
                            col2.metric("Filas duplicadas", f"{detalles_uni['filas_duplicadas']:,}")
                            col3.metric("Unicidad", f"{detalles_uni['pct_registros_unicos']:.1f}%")
                            
                            if detalles_uni['filas_duplicadas'] > 0:
                                st.warning(f"⚠️ Se detectaron **{detalles_uni['filas_duplicadas']}** filas duplicadas")
                            else:
                                st.success("✅ No hay filas duplicadas")
                            
                            if detalles_uni['columnas_con_duplicados_altos']:
                                st.warning("⚠️ **Columnas con baja variedad (<20% únicos):**")
                                df_dup = pd.DataFrame([
                                    {
                                        'Columna': col, 
                                        'Valores únicos': info['valores_unicos'],
                                        '% Unicidad': f"{info['pct_unicidad']:.1f}%"
                                    }
                                    for col, info in detalles_uni['columnas_con_duplicados_altos'].items()
                                ])
                                st.dataframe(df_dup, use_container_width=True, hide_index=True)
                        
                        with tab_prec:
                            detalles_prec = resultado_icd['detalles']['precision']
                            metodo_usado = detalles_prec.get('metodo_usado', 'iqr')
                            
                            # Mostrar método usado
                            metodo_nombre = {
                                'iqr': '📊 IQR (Cuartiles)',
                                'kmeans': '🎯 K-means',
                                'svm': '🤖 SVM (One-Class)',
                                'combinado': '🔄 Combinado (3 métodos)'
                            }.get(metodo_usado, metodo_usado)
                            
                            st.info(f"**Método usado para ICD:** {metodo_nombre}")
                            
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Datos numéricos", f"{detalles_prec['total_datos_numericos']:,}")
                            col2.metric("Outliers detectados", f"{detalles_prec['total_outliers']:,}")
                            col3.metric("Precisión", f"{detalles_prec['pct_datos_precisos']:.1f}%")
                            
                            if detalles_prec['outliers_por_columna']:
                                st.warning(f"⚠️ **Variables con outliers detectados:**")
                                
                                outliers_data = []
                                for col, info in detalles_prec['outliers_por_columna'].items():
                                    row = {
                                        'Variable': col,
                                        'Outliers': info['cantidad'],
                                        '% Outliers': f"{info['porcentaje']:.2f}%"
                                    }
                                    
                                    # Agregar columnas según el método
                                    if metodo_usado == 'iqr':
                                        row.update({
                                            'Límite Inferior': f"{info.get('limite_inferior', 0):.3f}",
                                            'Límite Superior': f"{info.get('limite_superior', 0):.3f}",
                                            'Valor Mín Atípico': f"{info.get('min_outlier', 0):.3f}",
                                            'Valor Máx Atípico': f"{info.get('max_outlier', 0):.3f}"
                                        })
                                    elif metodo_usado == 'kmeans':
                                        row.update({
                                            'Threshold': f"{info.get('threshold', 0):.3f}",
                                            'Valor Mín Atípico': f"{info.get('min_outlier', 0):.3f}",
                                            'Valor Máx Atípico': f"{info.get('max_outlier', 0):.3f}"
                                        })
                                    elif metodo_usado == 'svm':
                                        row.update({
                                            'Valor Mín Atípico': f"{info.get('min_outlier', 0):.3f}",
                                            'Valor Máx Atípico': f"{info.get('max_outlier', 0):.3f}"
                                        })
                                    elif metodo_usado == 'combinado':
                                        row.update({
                                            'Outliers IQR': info.get('outliers_iqr', 0),
                                            'Outliers K-means': info.get('outliers_kmeans', 0),
                                            'Outliers SVM': info.get('outliers_svm', 0),
                                            'Únicos': info.get('outliers_unicos', info['cantidad']),
                                            'Duplicados': info.get('overlapping', 0)
                                        })
                                    
                                    outliers_data.append(row)
                                
                                df_outliers = pd.DataFrame(outliers_data)
                                st.dataframe(df_outliers, use_container_width=True, hide_index=True)
                                
                                # Información del método
                                if metodo_usado == 'iqr':
                                    st.info("""
                                    **ℹ️ Método IQR:** Outliers = valores fuera de [Q1 - 1.5×IQR, Q3 + 1.5×IQR]
                                    Los outliers pueden ser valores legítimos extremos, no necesariamente errores.
                                    """)
                                elif metodo_usado == 'kmeans':
                                    st.info("""
                                    **ℹ️ Método K-means:** Agrupa datos y detecta puntos lejanos a centroides.
                                    Threshold basado en percentil 90 de distancias.
                                    """)
                                elif metodo_usado == 'svm':
                                    st.info("""
                                    **ℹ️ Método SVM:** Aprende frontera de distribución normal.
                                    Detecta anomalías con nu=0.1 (máx 10% outliers).
                                    """)
                                elif metodo_usado == 'combinado':
                                    st.info("""
                                    **ℹ️ Método Combinado:** Unión de outliers detectados por IQR, K-means y SVM.
                                    - **Únicos**: Outliers sin duplicar (un punto solo se cuenta una vez)
                                    - **Duplicados**: Outliers detectados por múltiples métodos
                                    - Proporciona detección exhaustiva considerando los 3 enfoques
                                    """)
                                
                                # Nueva sección: DataFrame completo con filas de outliers
                                st.markdown("---")
                                st.markdown("#### 📋 Filas Completas con Outliers")
                                
                                df_outliers_full = detalles_prec.get('df_outliers_completo', pd.DataFrame())
                                num_filas_outliers = detalles_prec.get('num_filas_con_outliers', 0)
                                
                                if not df_outliers_full.empty:
                                    st.markdown(f"**Total de filas con al menos un outlier: {num_filas_outliers}**")
                                    st.caption("Muestra las filas completas del dataset original que contienen valores atípicos")
                                    
                                    # Mostrar el dataframe con todas las columnas
                                    st.dataframe(
                                        df_outliers_full,
                                        use_container_width=True,
                                        height=400
                                    )
                                    
                                    # Opción para descargar
                                    csv_outliers = df_outliers_full.to_csv(index=True).encode('utf-8')
                                    st.download_button(
                                        label="📥 Descargar filas con outliers (CSV)",
                                        data=csv_outliers,
                                        file_name=f"outliers_completos_{metodo_usado}.csv",
                                        mime="text/csv",
                                        key="download_outliers_filas"
                                    )
                                else:
                                    st.info("No hay filas con outliers para mostrar")
                            else:
                                st.success("✅ No se detectaron outliers significativos")
                        
                        with tab_var:
                            detalles_var = resultado_icd['detalles']['variabilidad']
                            
                            col1, col2 = st.columns(2)
                            col1.metric("CV Promedio", f"{detalles_var['cv_promedio']:.1f}%")
                            col2.metric("% Variables CV adecuado", f"{detalles_var['pct_variabilidad_adecuada']:.1f}%")
                            
                            if detalles_var['cv_por_columna']:
                                st.markdown("**📊 Coeficiente de Variación:**")
                                
                                cv_data = []
                                for col, cv in detalles_var['cv_por_columna'].items():
                                    if abs(cv) < 10:
                                        categoria = "Baja"
                                        emoji_cv = "🟢"
                                    elif abs(cv) < 50:
                                        categoria = "Moderada"
                                        emoji_cv = "🟡"
                                    elif abs(cv) < 100:
                                        categoria = "Alta"
                                        emoji_cv = "🟠"
                                    else:
                                        categoria = "Muy Alta"
                                        emoji_cv = "🔴"
                                    
                                    cv_data.append({
                                        'Variable': col,
                                        'CV (%)': f"{cv:.2f}",
                                        'Categoría': f"{emoji_cv} {categoria}"
                                    })
                                
                                df_cv = pd.DataFrame(cv_data)
                                st.dataframe(df_cv, use_container_width=True, hide_index=True)
                                
                                if detalles_var['columnas_variabilidad_extrema']:
                                    st.warning("⚠️ **Variables con variabilidad extrema:**")
                                    for col, info in detalles_var['columnas_variabilidad_extrema'].items():
                                        st.write(f"- **{col}**: CV = {info['cv']:.1f}% - {info['problema']}")
                                
                                st.info("""
                                **ℹ️ Interpretación:** 🟢 <10% Baja | 🟡 10-50% Moderada | 🟠 50-100% Alta | 🔴 >100% Muy Alta
                                """)
                        
                        st.markdown("---")
                        
                        # Recomendaciones
                        st.markdown("### 💡 Recomendaciones")
                        recomendaciones = generar_recomendaciones(resultado_icd)
                        
                        for rec in recomendaciones:
                            st.markdown(rec)
                        
                        st.markdown("---")
                        
                        # Interpretación final
                        st.markdown("### 📝 Interpretación Final")
                        
                        if icd_total >= 90:
                            st.success(f"""
                            **{emoji} Excelente calidad ({icd_total:.1f}/100)**
                            
                            Los datos tienen calidad superior y están listos para análisis avanzados.
                            Mínimas correcciones necesarias.
                            """)
                        elif icd_total >= 75:
                            st.info(f"""
                            **{emoji} Buena calidad ({icd_total:.1f}/100)**
                            
                            Los datos son utilizables con limpieza menor. Revisar dimensiones con scores bajos.
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
                                    
                                    # Calcular correlaciones para tabla usando función centralizada
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
                                                use_container_width=True,
                                                hide_index=True
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
    if st.session_state.df is not None and openai_api_key:
        st.header("🤖 Agente de Análisis IA")
        
        # Función para crear/recrear el agente
        def create_agent():
            llm = ChatOpenAI(
                model=model_name,
                temperature=temperature,
                openai_api_key=openai_api_key
            )
            return create_pandas_dataframe_agent(
                llm,
                st.session_state.df,
                verbose=False,  # Cambiar a False para evitar output duplicado
                agent_type=AgentType.OPENAI_FUNCTIONS,
                allow_dangerous_code=True
            )
        
        # Inicializar o recrear agente solo cuando sea necesario
        # Verificar si necesitamos recrear el agente (cambio de modelo, temperatura o datos)
        agent_config_key = f"{model_name}_{temperature}_{id(st.session_state.df)}"
        
        if 'agent_config_key' not in st.session_state:
            st.session_state.agent_config_key = None
        
        if st.session_state.agent is None or st.session_state.agent_config_key != agent_config_key:
            try:
                st.session_state.agent = create_agent()
                st.session_state.agent_config_key = agent_config_key
            except Exception as e:
                st.error(f"❌ Error al inicializar el agente: {str(e)}")
                st.info("Verifica que tu API key de OpenAI sea válida y tenga créditos disponibles.")
                st.session_state.agent = None
        
        if st.session_state.agent is not None:
            st.success("🎯 Agente IA inicializado correctamente")
            
            # Ejemplos de preguntas
            st.subheader("💡 Ejemplos de preguntas que puedes hacer:")
            examples = [
                "¿Cuántas filas tiene el dataset?",
                "¿Cuáles son las columnas numéricas?",
                "Muestra un resumen estadístico de los datos",
                "¿Hay valores nulos en el dataset?",
                "¿Cuál es la correlación entre las variables numéricas?",
                "¿Cuáles son los valores únicos de [nombre_columna]?",
                "Calcula la media, mediana y moda de [columna_numerica]",
                "¿Qué cultivos se dan en cereté?"
            ]
            
            for i, example in enumerate(examples, 1):
                st.write(f"{i}. {example}")
            
            # Interface para hacer preguntas
            st.subheader("❓ Haz tu pregunta sobre los datos")
            
            # Usar un formulario para evitar rerun automático
            with st.form(key="question_form", clear_on_submit=True):
                user_question = st.text_input(
                    "Escribe tu pregunta:",
                    placeholder="Ej: ¿Cuál es la correlación entre las variables numéricas?",
                    key="user_input_form"
                )
                
                col1, col2 = st.columns([1, 4])
                with col1:
                    ask_button = st.form_submit_button("🚀 Preguntar", type="primary")
                with col2:
                    pass  # Espacio vacío para mantener el layout
            
            # Botón de limpiar historial fuera del formulario
            if st.button("🗑️ Limpiar historial"):
                st.session_state.chat_history = []
                st.rerun()
            
            if ask_button and user_question:
                with st.spinner("🔄 El agente está analizando tus datos..."):
                    try:
                        # Ejecutar la pregunta con el agente
                        response = st.session_state.agent.invoke({"input": user_question})
                        
                        # Agregar al historial
                        st.session_state.chat_history.append({
                            "question": user_question,
                            "answer": response["output"]
                        })
                        
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
    
    else:
        if st.session_state.df is None:
            st.info("📁 Por favor carga datos primero en la pestaña 'Cargar Datos'")
        else:
            st.warning("⚠️ Por favor ingresa tu API Key de OpenAI en la barra lateral")

# Información inicial
if st.session_state.df is None:
    st.divider()
    st.info("""
    ### 🚀 Cómo usar:
    
    **Opción 1: Archivo CSV/Excel**
    1. 📁 Sube un archivo CSV o Excel
    2. 🧹 Opcional: Aplica limpieza automática
    
    **Opción 2: API Socrata**
    1. 🌍 Ingresa dominio y Dataset ID
    2. 🔄 Carga desde API
    3. 🧹 Opcional: Aplica limpieza automática
    
    **Luego:**
    - 📊 Consulta estadísticos y calidad de datos
    - 📈 Ve visualizaciones y correlaciones
    - 🤖 Haz preguntas en lenguaje natural con IA
    """)

# Footer
st.divider()
st.caption("📊 Agente datos suelos Agrosavia | Análisis con OpenAI GPT e Índice de Calidad de Datos")
