import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from distribuciones_discretas import Binomial

# Configuración inicial de la interfaz web
st.set_page_config(page_title="Simulador Binomial", page_icon="🧪", layout="wide")

st.title("🧪 Simulador Interactivo de Distribución Binomial")
# ─── SECCIÓN PEDAGÓGICA Y GUÍA DE USUARIO (DESPLEGABLE) ────────────────────
with st.expander("📖 ¿Qué es la Distribución Binomial y para qué sirve? (Guía Rápida)", expanded=True):
    st.markdown("""
    ### 🔬 ¿Por qué y para qué se utiliza?
    La **Distribución Binomial** es uno de los modelos matemáticos más utilizados en ciencias de la salud, control de calidad farmacéutico y bioestadística. Se usa para calcular la probabilidad de éxito o fracaso en eventos que tienen una respuesta **binaria** (solo dos opciones posibles):
    * Un medicamento surte efecto o no lo hace.
    * Un lote de tabletas pasa el control de calidad o sale defectuoso.
    * Una prueba diagnóstica da resultado positivo o negativo.
    
    ### ⚙️ ¿Cómo utilizar este simulador?
    1. **Define tu escenario (Barra lateral izquierda):**
        * **Número de ensayos ($n$):** Es el tamaño del grupo o muestra que vas a evaluar (ej. cantidad de pacientes evaluados o blísteres revisados).
        * **Probabilidad de éxito ($p$):** El porcentaje histórico o esperado de que ocurra el evento de interés, ingresado en decimales (ej. un $5\\%$ de error se escribe como `0.05`).
        * **Número de simulaciones ($N$):** Cuántas veces la computadora recreará el experimento usando el **Método de Monte Carlo** para contrastar la teoría matemática con la realidad experimental.
        
    2. **Interpreta los resultados en pantalla:**
        * **Métricas estadísticas:** Revisa si la media y la varianza que arroja la teoría matemática coinciden con los datos simulados de forma aleatoria.
        * **Gráficas y Tabla de Datos (Abajo):** Busca en la fila del número de éxitos exactos (**$k$**) que necesitas responder:
            * Mira la columna **`P(X=k) Teoría`** si te preguntan por un valor *exacto* (ej. "probabilidad de que se curen exactamente 3 pacientes").
            * Mira la columna **`P(X<=k) Teoría`** si te preguntan por un valor *acumulado* (ej. "probabilidad de encontrar como máximo 2 piezas defectuosas").
    """)

st.markdown("---")
# ─── 1. BARRA LATERAL PARA CONTROLES INTERACTIVOS ─────────────────────────────
st.sidebar.header("⚙️ Parámetros de la Simulación")

n = st.sidebar.slider("Número de ensayos (n)", min_value=1, max_value=100, value=10, step=1)
p = st.sidebar.slider("Probabilidad de éxito (p)", min_value=0.0, max_value=1.0, value=0.3, step=0.01)
N = st.sidebar.number_input("Número de simulaciones (N)", min_value=100, max_value=100000, value=10000, step=500)

# ─── 2. PROCESAMIENTO LÓGICO ──────────────────────────────────────────────────
np.random.seed(42)
b = Binomial(n=n, p=p)
sim = np.random.binomial(n, p, N)

# ─── 3. METRICAS ESTADÍSTICAS EN PANTALLA ─────────────────────────────────────
st.subheader("📊 Comparación de Parámetros Estadísticos")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="Media Teórica", value=f"{b.media():.4f}")
    st.metric(label="Media Simulada", value=f"{sim.mean():.4f}", delta=f"{sim.mean() - b.media():.4f}")

with col2:
    st.metric(label="Varianza Teórica", value=f"{b.varianza():.4f}")
    st.metric(label="Varianza Simulada", value=f"{sim.var():.4f}", delta=f"{sim.var() - b.varianza():.4f}")

with col3:
    st.metric(label="Desv. Estándar Teórica", value=f"{b.desviacion_estandar():.4f}")
    st.metric(label="Desv. Estándar Simulada", value=f"{sim.std():.4f}", delta=f"{sim.std() - b.desviacion_estandar():.4f}")

st.markdown("---")

# ─── SECCIÓN DE INTERPRETACIÓN AUTOMATIZADA EN TIEMPO REAL ──────────────────
st.markdown("### 📝 Análisis Automatizado de los Resultados")

# Guardamos los valores teóricos en variables simples para el texto
media_valor = n * p
varianza_valor = n * p * (1 - p)

# Creamos un recuadro informativo con la conclusión analítica
with st.container():
    st.info(f"""
    **Análisis de Tendencia Central y Variabilidad:**
    * **¿Qué significa la Media ({media_valor:.2f})?:** Si repitiéramos este experimento con grupos de **{n}** individuos de forma indefinida, el resultado promedio más esperado es encontrar exactamente **{media_valor:.2f}** casos con éxito. 
    * **¿Qué significa la Varianza ({varianza_valor:.2f})?:** Nos indica qué tan dispersos o agrupados están los datos reales respecto al promedio. Al ser una varianza de **{varianza_valor:.2f}**, los resultados de los experimentos individuales no se alejarán drásticamente de la media.
    
    **Conclusión del Escenario Evaluado:**
    El modelo binomial demuestra que al tener una probabilidad de éxito del **{p*100:.1f}%** en una muestra de **{n}** ensayos, el comportamiento de la distribución se concentrará firmemente alrededor de **{int(media_valor)}** éxitos. Cualquier valor observado en la práctica que esté muy alejado de este rango teóricamente esperado indicaría un comportamiento atípico o una anomalía en el lote o muestra bajo análisis.
    """)

st.markdown("---")
# ─── 4. RENDERIZADO DE GRÁFICAS EN TIEMPO REAL ────────────────────────────────
st.subheader("📈 Gráficas Distribucionales")

ks = np.arange(0, n+1)
pmf_t = b.pmf(ks)
pmf_s = np.array([(sim==k).mean() for k in ks])
cdf_t = b.cdf(ks)
cdf_s = np.array([(sim<=k).mean() for k in ks])

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(f"Binomial (n={n}, p={p})  —  Teoría vs Simulación (N={N})", fontsize=13)

# Gráfica PMF
axes[0].bar(ks-0.2, pmf_t, 0.4, label="Teoría", color="#3B82F6", alpha=0.85)
axes[0].bar(ks+0.2, pmf_s, 0.4, label="Simulación", color="#F59E0B", alpha=0.85)
axes[0].axvline(b.media(), color="red", linestyle="--", linewidth=1.5, label=f"Media Teórica={b.media():.2f}")
axes[0].set_xlabel("k (Éxitos)")
axes[0].set_ylabel("P(X = k)")
axes[0].set_title("Función de Masa de Probabilidad (PMF)")
axes[0].legend()
axes[0].grid(axis="y", alpha=0.3)

# Gráfica CDF
axes[1].bar(ks-0.2, cdf_t, 0.4, label="Teoría", color="#3B82F6", alpha=0.85)
axes[1].bar(ks+0.2, cdf_s, 0.4, label="Simulación", color="#F59E0B", alpha=0.85)
axes[1].set_xlabel("k (Éxitos)")
axes[1].set_ylabel("P(X <= k)")
axes[1].set_title("Probabilidad Acumulada (CDF)")
axes[1].legend()
axes[1].grid(axis="y", alpha=0.3)

plt.tight_layout()
st.pyplot(fig)

st.markdown("---")

# ─── 5. TABLA DE DATOS INTERACTIVA ───────────────────────────────────────────
st.subheader("📋 Tabla de Datos de Distribución")

datos_tabla = []
for k in range(n+1):
    datos_tabla.append({
        "k": k,
        "P(X=k) Teoría": f"{b.pmf(k):.4f}",
        "P(X=k) Simulada": f"{(sim==k).mean():.4f}",
        "P(X<=k) Teoría": f"{b.cdf(k):.4f}",
        "P(X<=k) Simulada": f"{(sim<=k).mean():.4f}"
    })

df_resultados = pd.DataFrame(datos_tabla)
st.dataframe(df_resultados, use_container_width=True)
