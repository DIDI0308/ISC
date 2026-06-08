import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix

st.set_page_config(page_title="Proyecto Final", layout="wide")

st.markdown("""
<style>
    .main {
        background-color: #121212;
        color: #E0E0E0;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    h1, h2, h3, h4 {
        color: #FF4B4B !important;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .st-expander {
        background-color: #1E1E1E !important;
        border: 1px solid #333 !important;
        border-radius: 8px !important;
    }
    div[data-testid="stMetric"] {
        background-color: #1E1E1E;
        border-left: 4px solid #FF4B4B;
        border-radius: 4px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    hr {
        border-color: #333;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1E1E1E;
        border-radius: 4px 4px 0 0;
        border: 1px solid #333;
        border-bottom: none;
        padding: 10px 20px;
        color: #E0E0E0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FF4B4B !important;
        color: #FFFFFF !important;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center;'>VOC INSIGHTS ANALYTICS: PREDICCIÓN DE DETRACTORES Y ANÁLISIS DE ATRIBUTOS CRÍTICOS EN EL ISC</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #E0E0E0 !important;'>PROYECTO FINAL</h3>", unsafe_allow_html=True)
st.markdown("---")

tab_obj, tab_prob, tab_eda, tab_prep, tab_mod, tab_eval, tab_pred, tab_conc = st.tabs([
    "Objetivo", 
    "Problema", 
    "EDA", 
    "Preprocesamiento", 
    "Modelamiento", 
    "Evaluación", 
    "Predicción", 
    "Conclusiones"
])

with tab_obj:
    with st.expander("Objetivo del Proyecto", expanded=True):
        st.write("""
        El objetivo principal de este proyecto es desarrollar e implementar un modelo de Machine Learning que permita mejorar la gestión de la calidad en el servicio de posventa en Taiyo Motors S.A.. Por un lado, se busca anticipar la probabilidad de que un cliente se convierta en detractor antes del cierre mensual, de modo que el equipo pueda actuar a tiempo con medidas de retención y no cuando el impacto ya se refleja en el indicador. Por otro lado, el modelo pretende ir más allá y ayudar a entender qué está fallando realmente en la operación, identificando el peso que tienen atributos como la recepción, los tiempos de espera, la limpieza o la claridad en las explicaciones del asesor sobre la satisfacción del cliente. Es asi que este enfoque permite convertir los datos en una herramienta útil para la toma de decisiones al detectar clientes en riesgo y para señalar con claridad dónde enfocar los esfuerzos de mejora continua en línea con la lógica de trabajo de Kaizen.
        """)

with tab_prob:
    with st.expander("Planteamiento del Problema", expanded=True):
        st.write("""
        El problema se origina en la forma en que actualmente se mide el Índice de Satisfacción del Cliente ISC dentro de Taiyo Motors S.A., empresa del sector automotriz encargada de la comercialización y servicio de posventa de vehículos de la marca Nissan. En el área de posventa, el ISC se utiliza como principal indicador para evaluar la satisfacción del cliente, sin embargo, su naturaleza es completamente reactiva, ya que se analiza al cierre del mes, cuando la experiencia del cliente ya ocurrió y no existe posibilidad de intervención y no hay cumplimiento del objetivo.
        
        A esta limitación se suma la fórmula de cálculo, la cual penaliza doble a las calificaciones iguales o menores a 6. Esto implica que un solo cliente insatisfecho puede contrarrestar el impacto positivo de múltiples atenciones satisfactorias, generando una alta volatilidad en el indicador y dificultando la gestión real de la calidad del servicio.
        
        Es asi que surge la necesidad de evolucionar hacia un enfoque más avanzado, que permita anticipar el comportamiento del cliente antes de que se materialice en una calificación negativa. Pasar de una medición retrospectiva a un sistema predictivo y prescriptivo que no solo identifique qué clientes tienen mayor probabilidad de convertirse en detractores, sino que también permita entender qué fallas específicas en el servicio, como tiempos de atención, calidad del trabajo o comunicación del asesor, están generando dicha insatisfacción. De esta manera, se busca habilitar intervenciones oportunas dentro del servicio de posventa, alineadas con la filosofía Kaizen, con el objetivo de mejorar de forma sostenida la satisfacción del cliente y la calidad del servicio ofrecido.
        """)

URL_MAESTRA = "https://docs.google.com/spreadsheets/d/1OsTGidjydtPS6kJGhZCTQTXFxrrHxrq5G0MrYQ5KkDM/export?format=csv"

@st.cache_data(ttl=3600, show_spinner=False)
def cargar_datos_crudos():
    return pd.read_csv(URL_MAESTRA, low_memory=False)

with st.spinner("Procesando entorno..."):
    df_crudo = cargar_datos_crudos()

with tab_eda:
    with st.expander("Entendimiento de los Datos", expanded=True):
        st.write("""
        La materia prima proviene de una base alojada en la nube, donde se compilaron las encuestas desde abril de 2025. Se revela una base con alta dimensionalidad al contar con 68 columnas. La mayoría de los clientes otorga notas altas, dejando a los detractores como una anomalía estadística, es decir que son menos del 5%. Esto dicta que no podemos usar modelos estándar, pues estos simplemente "aprenderían" a predecir que todos los clientes están felices, volviéndose inútiles para el análisis.
        """)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Volumen de Registros", df_crudo.shape[0])
        col2.metric("Dimensionalidad (Columnas)", df_crudo.shape[1])
        col3.metric("Origen de Datos", "Google Sheets API")
        
        st.markdown("#### Análisis Visual Exploratorio")
        col_voc_temp = [col for col in df_crudo.columns if 'VOC01' in col][0]
        df_viz = df_crudo.dropna(subset=[col_voc_temp]).copy()
        df_viz[col_voc_temp] = pd.to_numeric(df_viz[col_voc_temp], errors='coerce')
        
        plt.style.use('dark_background')
        fig_eda, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), facecolor='#121212')
        
        sns.countplot(x=col_voc_temp, data=df_viz, palette='Reds', ax=ax1)
        ax1.set_title("Distribución de Calificaciones Generales", color='#FF4B4B')
        ax1.set_xlabel("Nota de Satisfacción (1-10)", color='white')
        ax1.set_ylabel("Cantidad de Clientes", color='white')
        ax1.tick_params(colors='white')
        
        for p in ax1.patches:
            height = p.get_height()
            if height > 0:
                ax1.text(p.get_x() + p.get_width()/2., height + 5,
                        f'{int(height)}', ha="center", va="bottom", color='white', fontweight='bold', fontsize=9)
        
        df_viz['Clase'] = np.where(df_viz[col_voc_temp] <= 6, 'Detractor (<=6)', 'Satisfecho (>6)')
        colores_torta = ['#FF4B4B', '#FFCDD2']
        
        df_clases = df_viz['Clase'].value_counts()
        wedges, texts, autotexts = ax2.pie(df_clases, labels=df_clases.index, autopct='%1.1f%%', colors=colores_torta, startangle=90)
        
        for autotext, wedge in zip(autotexts, wedges):
            r, g, b, _ = wedge.get_facecolor()
            luminance = 0.299*r + 0.587*g + 0.114*b
            autotext.set_color('black' if luminance > 0.5 else 'white')
            autotext.set_fontweight('bold')
        
        for text in texts:
            text.set_color('white')
            
        ax2.set_ylabel('')
        ax2.set_title("Proporción de Riesgo", color='#FF4B4B')
        
        st.pyplot(fig_eda)

@st.cache_data(show_spinner=False)
def preprocesar_pipeline(df):
    df_temp = df.copy()
    
    col_voc01 = [col for col in df_temp.columns if 'VOC01' in col][0]
    col_voc08 = [col for col in df_temp.columns if 'VOC08' in col][0]
    
    df_temp[col_voc08] = pd.to_numeric(df_temp[col_voc08], errors='coerce')
    df_temp = df_temp[df_temp[col_voc08] == 1]
    
    df_temp[col_voc01] = pd.to_numeric(df_temp[col_voc01], errors='coerce')
    df_temp = df_temp[(df_temp[col_voc01] >= 1) & (df_temp[col_voc01] <= 10)]
    
    mapeo_columnas = {
        32: 'Recepcion_vehiculo', 34: 'Facilidad_cita', 35: 'Rapidez_recepcion',
        36: 'Disposicion_asesor', 37: 'Explicacion_trabajo_pre', 40: 'Instalaciones',
        41: 'Entrega_vehiculo', 42: 'Tiempo_espera_entrega', 43: 'Limpieza_vehiculo',
        44: 'Explicacion_trabajo_post', 45: 'Explicacion_costo', 46: 'Calidad_trabajo',
        47: 'Valor_pagado', 48: 'Puntualidad_fecha', 49: 'Puntualidad_hora',
        51: 'Tiempo_agencia', 53: 'Atencion_personal', 54: 'Amabilidad_personal',
        56: 'Seguimiento_asesor'
    }
    
    indices_a_usar = list(mapeo_columnas.keys())
    nombres_nuevos = list(mapeo_columnas.values())
    
    df_features = df_temp.iloc[:, indices_a_usar].copy()
    df_features.columns = nombres_nuevos
    df_features['Target_VOC01'] = df_temp[col_voc01].values
    
    umbral = 0.7 * len(df_features)
    columnas_validas = []
    
    for col in nombres_nuevos:
        if df_features[col].isnull().sum() <= umbral:
            columnas_validas.append(col)
            
    df_final = df_features[columnas_validas + ['Target_VOC01']].copy()
    df_final['Es_Detractor'] = np.where(df_final['Target_VOC01'] <= 6, 1, 0)
    
    for col in columnas_validas:
        df_final[col] = pd.to_numeric(df_final[col], errors='coerce')
        df_final[col] = df_final[col].fillna(df_final[col].median())
        
    return df_final, columnas_validas

df_modelo, cols_validas = preprocesar_pipeline(df_crudo)

with tab_prep:
    with st.expander("Preprocesamiento", expanded=True):
        st.write("""
        Para que los algoritmos no aprendan con ruido, se aplicó el siguiente preprocesamiento:
        
        1. Se descartaron las encuestas donde la variable VOC08 indicaba que el cliente no deseaba continuar, asegurando que el modelo solo aprenda de experiencias completas. 
        2. Se transformaron los códigos crudos de los encabezados de las columnas en atributos legibles, ajustando el desfase de índices (+1) entre la vista y la matriz de Python.
        3. Las variables con más de un 70% de datos faltantes fueron eliminadas para evitar sesgos. 
        4. Para el resto de las encuestas incompletas, se aplicó una imputación estocástica por la mediana, preservando la distribución original sin distorsionar la realidad del servicio.
        """)
        st.dataframe(df_modelo.head())

X = df_modelo[cols_validas]
y = df_modelo['Es_Detractor']

if y.sum() >= 2:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    with st.spinner("Construyendo arquitecturas..."):
        lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
        lr.fit(X_train, y_train)

        rf = RandomForestClassifier(class_weight='balanced', n_estimators=150, max_depth=6, random_state=42)
        rf.fit(X_train, y_train)

        ratio_desb = len(y_train[y_train == 0]) / max(1, len(y_train[y_train == 1]))
        pesos = np.where(y_train == 1, ratio_desb, 1.0)
        
        gbc = GradientBoostingClassifier(n_estimators=150, max_depth=4, random_state=42)
        gbc.fit(X_train, y_train, sample_weight=pesos)

with tab_mod:
    with st.expander("Modelamiento", expanded=True):
        st.write("""
        En esta etapa se construyo una estrategia que permitió al modelo aprender correctamente a partir de una base de datos desbalanceada, donde los casos críticos de los clientes detractores son muy pocos en comparación con los satisfechos.
        
        Primero, se realizó la partición de los datos en conjuntos de entrenamiento 80% y prueba 20% utilizando la función train_test_split. Sin embargo, esta partición no se hizo aleatoriamente, se incorporó el parámetro stratify=y porque, al existir una proporción baja de detractores frente a la mayoría satisfechos, se podría generar un conjunto de prueba sin ningún caso con detractores, lo que haría imposible evaluar los modelos. Al estratificar, se aseguró que la proporción de detractores y satisfechos sea consistente.
        
        1. Como primer modelo, se usó la Regresión Logística, que es como una línea base de comparación. Se configuró con el parámetro class_weight='balanced', lo que ayuda a compensar el desbalance de las clases y así el modelo penaliza mucho más los errores cometidos sobre detractores que sobre los clientes satisfechos para prestar mayor atención a estos casos.
        
        2. Como segundo modelo, se usó Random Forest, con 150 árboles de decisión y con una profundidad de 6 niveles para evitar el overfitting. Este modelo permitio capturar relaciones que son más complejas entre las variables, ya que cada árbol generó reglas basadas en las combinaciones de los atributos. Pudo identificar situaciones donde una buena experiencia en recepción se ve afectada por las fallas en la entrega o en la limpieza del vehículo. Al trabajar con muchos árboles y promediar sus resultados, se obtiene una predicción mucho más robusta.
        
        3. Como tercer modelo se aplicó Gradient Boosting, el cual trabaja de manera secuencial, cada nuevo árbol se construye corrigiendo los errores del anterior árbol. Este modelo no incorpora un parámetro de balanceo de clases, por lo que si fue necesario calcular manualmente los pesos para cada observación con sample_weight, De esta manera, se logró que el modelo pueda enfocar su aprendizaje en los casos donde más falla, osea en los detractores, ajustando asi progresivamente su capacidad predictiva en cada iteración.
        """)

with tab_eval:
    with st.expander("Evaluación", expanded=True):
        st.write("""
        En esta etapa, el foco estuvo en evaluar qué tan útil eran los modelos para el problema que enfrenta Taiyo Motors. Se dejó de lado el uso del accuracy como la métrica principal, ya que con datos desbalanceados puede resultar engañoso. En este caso, más del 97% de los clientes son satisfechos, por lo que un modelo que solo clasifique a todos como satisfechos alcanzaría un alto nivel de exactitud, pero no tendría ningún valor ya que no detectaría a los clientes que están en riesgo.
        
        Por esto la evaluación se centró en analizar la matriz de confusión y en la capacidad del modelo para identificar bien a los detractores. A través del classification report y la visualización de la matriz, se pudo analizar con mayor detalle los errores de cada modelo.
        
        La interpretación de los errores se hizo según la lógica de falsos positivos, lo que implica que el modelo alerta sobre un posible cliente insatisfecho cuando en realidad no lo está, lo cual tiene un costo bajo, limitado a una acción preventiva como una llamada de seguimiento y también según los falsos negativos, que representan a un cliente que el modelo considera satisfecho, pero que en realidad termina siendo un detractor y este tipo de error es más crítico, ya que implica la pérdida potencial del cliente y de su valor a largo plazo.
        
        Según esto, se priorizó el modelo con mayor capacidad para reducir los falsos negativos. El modelo de Gradient Boosting mostró el mejor desempeño, ya que el enfoque secuencial y el uso de pesos para manejar el desbalance de las clases, le permitió mejorar en la identificación de los detractores. Por eso, se le seleccionó como el modelo final, al ofrecer el mejor equilibrio entre el desempeño y la relevancia para tomar decisiones.
        """)
        
        if y.sum() >= 2:
            def mostrar_reporte_y_matriz(modelo):
                preds = modelo.predict(X_test)
                col_m1, col_m2 = st.columns([1.5, 1])
                with col_m1:
                    reporte = classification_report(y_test, preds, output_dict=True)
                    df_rep = pd.DataFrame(reporte).transpose()
                    st.dataframe(df_rep.style.format("{:.2f}").highlight_max(axis=0, color='#FF4B4B'))
                with col_m2:
                    cm = confusion_matrix(y_test, preds)
                    fig_cm, ax_cm = plt.subplots(figsize=(4, 3), facecolor='#121212')
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', cbar=False, 
                                xticklabels=['Satisfecho', 'Detractor'], 
                                yticklabels=['Satisfecho', 'Detractor'],
                                annot_kws={"color": "white", "weight": "bold"})
                    ax_cm.set_ylabel('Realidad', color='white')
                    ax_cm.set_xlabel('Predicción del Modelo', color='white')
                    ax_cm.tick_params(colors='white')
                    st.pyplot(fig_cm)

            t_gb, t_rf, t_lr = st.tabs(["Gradient Boosting", "Random Forest", "Regresión Logística"])
            with t_gb:
                mostrar_reporte_y_matriz(gbc)
            with t_rf:
                mostrar_reporte_y_matriz(rf)
            with t_lr:
                mostrar_reporte_y_matriz(lr)

with tab_pred:
    with st.expander("Predicción y Extracción de Conocimiento", expanded=True):
        st.write("""
        En esta etapa, se trabajó en tener una herramienta que puede ser usada en la gestión para generar información para tomar decisiones. En lugar de usar la función predict(), ya que solo clasifica a los clientes como detractores o no detractores, se uso predict_proba(), lo que permitió obtener una probabilidad asociada a cada predicción y en lugar tener una respuesta cerrada, se obtiene un nivel de riesgo. En vez de indicar que un cliente será detractor, se puede ver que tiene un 83% de probabilidad de serlo y esta información permite trazar las acciones, para estos casos que tienen mayor riesgo limitando los recursos a usar.
        
        Tambien se buscó extraer el conocimiento del modelo para entender las variables que están influyendo en las predicciones. Para esto, se uso feature_importances_ del modelo de Gradient Boosting, distribuyendo su capacidad predictiva entre las variables de entrada. Según esto, se identificaron los atributos que tienen mayor impacto en la probabilidad de que un cliente se convierta en detractor.
        """)
        
        if y.sum() >= 2:
            importancias = pd.Series(gbc.feature_importances_, index=X.columns).sort_values(ascending=False)
            col_p1, col_p2 = st.columns([1, 2])
            
            with col_p1:
                st.dataframe(importancias.head(5).to_frame(name="Peso").style.format("{:.1%}"))
            
            with col_p2:
                fig_feat, ax_feat = plt.subplots(figsize=(8, 4), facecolor='#121212')
                sns.barplot(x=importancias.head(5).values, y=importancias.head(5).index, palette='Reds_r', ax=ax_feat)
                ax_feat.set_title("Importancia de Atributos", color='#FF4B4B')
                ax_feat.set_xlabel("Impacto Relativo", color='white')
                ax_feat.tick_params(colors='white')
                
                for p in ax_feat.patches:
                    width = p.get_width()
                    r, g, b, _ = p.get_facecolor()
                    luminance = 0.299*r + 0.587*g + 0.114*b
                    text_color = 'black' if luminance > 0.5 else 'white'
                    ax_feat.text(width/2, p.get_y() + p.get_height()/2., 
                                 f'{width*100:.1f}%', ha="center", va="center", color=text_color, fontweight='bold')
                                 
                st.pyplot(fig_feat)

with tab_conc:
    with st.expander("Conclusiones y Recomendaciones", expanded=True):
        st.write("""
        El despliegue del portal permite demostrar que es viable automatizar el análisis del VoC. Sin embargo, al evaluar el desempeño del modelo, aparece una limitación importante que es el desbalanceo de la muestra de los datos. En la base analizada, existe una mayoría de clientes satisfechos, lo que hace que la clase de clientes de interés sea muy reducida para el entrenamiento de los modelos predictivos. 
        
        El uso de Gradient Boosting junto con la asignación de pesos a las observaciones ayudó a reducir este sesgo y el modelo alcanzó un recall del 25% en el conjunto de prueba. Este resultado confirma que la lógica es correcta pero el sistema necesita un proceso de maduración. A medida que se incorporen nuevos datos mensuales, el modelo tendrá más data de detractores y su capacidad predictiva mejorará de forma progresiva. 
        
        El valor del proyecto se encuentra en el diagnóstico que permite realizar. El análisis de importancia de las variables muestra con claridad que la insatisfacción del cliente no se origina en fallas del vehículo, sino en la experiencia inicial de atención. Las variables con mayor peso en la predicción del riesgo de tener detractores están concentradas en la etapa de recepción y en la interacción con el asesor de servicio. En particular, destacan la explicación previa del trabajo a realizar con el 21.0%, la rapidez en la recepción con el 17.6%, la disposición del asesor para escuchar con el 16.3% y el proceso de recepción con el 14.8%. En conjunto, estos atributos explican cerca del 70% del riesgo total que hay de que un cliente califique con una nota igual o menor a 6 y esto sugiere que la percepción del servicio se define en el primer contacto, antes de que el vehículo ingrese al taller.
        
        A partir de esto, se recomienda que la mejora se concentre en la recepción. Es clave estandarizar este proceso poniéndole énfasis en la comunicación y en las habilidades del asesor para para cumplir y superar las expectativas del cliente. Es asi que el portal elaborado en Streamlit debería mantenerse como una herramienta activa, alimentándose continuamente con nuevos datos, lo que fortalecerá el modelo y mejorará su detección. Adicionalmente, considerando que un detractor tiene alto impacto en el ISC, pueden aprovecharse oportunidades de contacto preventivo como una llamada de seguimiento.
        
        Finalmente, como propuesta de mejora futura, es valioso incorporar una segmentación más detallada del análisis por taller y en una siguiente etapa a nivel de asesor de servicio ya que esto permitiría identificar las diferencias en el desempeño entre los talleres y las personas, facilitando las intervenciones alineados con la mejora continua.
        """)