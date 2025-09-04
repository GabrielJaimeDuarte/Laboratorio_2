Práctica 2 – Laboratorio

Curso: Inteligencia Artificial / Deep Learning
Integrantes: [Camilo Cifuentes y Gabriel Duarte]
Fecha: [31/08/2025]

Punto 1 – Investigación de Librerías  

En la siguiente práctica se trabajará con un ecosistema de librerías de **ciencia de datos, visualización y despliegue**.  

---

# 1. Librerías de datos  
| Librería   | Propósito | Ejemplo mínimo |
|------------|-----------|----------------|
| **Pandas** | Manejo de datos tabulares (DataFrames). | ```python<br>import pandas as pd<br>df = pd.DataFrame({"A":[1,2],"B":[3,4]})<br>print(df.head())``` |
| **Polars** | Alternativa a Pandas, optimizada en Rust para mayor velocidad. | ```python<br>import polars as pl<br>df = pl.DataFrame({"A":[1,2],"B":[3,4]})``` |
| **Dask** | Permite procesar datasets grandes en paralelo. | ```python<br>import dask.dataframe as dd<br>df = dd.read_csv("*.csv")``` |
| **GeoPandas** | Extiende Pandas para análisis geoespacial. | ```python<br>import geopandas as gpd<br>world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))``` |
| **Xarray** | Datos multidimensionales (clima, imágenes, señales). | ```python<br>import xarray as xr<br>data = xr.DataArray([[1,2],[3,4]])``` |
| **DuckDB** | Motor SQL embebido para consultas rápidas. | ```python<br>import duckdb<br>duckdb.sql("SELECT 42").show()``` |
| **Ibis** | Conector SQL unificado (BigQuery, DuckDB, etc.). | ```python<br>import ibis<br>con = ibis.sqlite.connect("mi.db")``` |
| **RAPIDS** | Aceleración en GPU (NVIDIA) para datos grandes. | ```python<br>import cudf<br>gdf = cudf.DataFrame({"a":[1,2,3]})``` |
| **Intake** | Manejo de catálogos de datasets. | ```python<br>import intake<br>cat = intake.open_catalog("catalog.yml")``` |
| **NetworkX** | Creación y análisis de grafos/redes. | ```python<br>import networkx as nx<br>G = nx.Graph(); G.add_edge("A","B")``` |

---

# 2. Librerías de representación intermedia  
| Librería   | Propósito | Ejemplo mínimo |
|------------|-----------|----------------|
| **hvPlot** | API unificada para graficar (`.hvplot`). | ```python<br>import pandas as pd, hvplot.pandas<br>df = pd.DataFrame({"x": range(5), "y":[i**2 for i in range(5)]})<br>df.hvplot.line(x="x", y="y")``` |
| **HoloViews** | Representación de alto nivel para visualización automática. | ```python<br>import holoviews as hv<br>hv.Curve([(0,0),(1,1),(2,4)])``` |
| **Datashader** | Renderiza millones de puntos de forma eficiente. | ```python<br>import datashader as ds``` |

---

# 3. Librerías de salida y despliegue  
| Librería   | Propósito |
|------------|-----------|
| **Bokeh** | Gráficos interactivos en la web. |
| **Matplotlib** | Gráficos estáticos 2D/3D, base de la mayoría. |
| **Plotly** | Gráficos interactivos, soporte 3D y dashboards. |
| **Streamlit** | Framework para crear aplicaciones web interactivas con Python. |

Ejemplo con **Streamlit**:
```python
import streamlit as st
st.title("Hola Streamlit 👋")
st.line_chart({"y":[1,2,3,4,5]})
```
Punto 2 – Conceptos Clave

En este punto se presentan tres conceptos fundamentales relacionados con el modelo de agentes y el control basado en campos de potenciales.

---

1. Agente Inteligente
Un **agente inteligente** es una entidad autónoma (software o hardware) capaz de **percibir su entorno** mediante sensores, **razonar** sobre esa información, y **actuar** en consecuencia para alcanzar objetivos definidos.  
Sus características principales son:  
- Autonomía: toma decisiones sin intervención externa.  
- Adaptabilidad: ajusta su comportamiento ante cambios en el entorno.  
- Proactividad: persigue metas específicas, no solo responde a estímulos.  
- Capacidad de comunicación: puede interactuar con otros agentes.  

Ejemplo: Un robot móvil que percibe obstáculos con sensores ultrasónicos, planifica su ruta y se mueve para llegar a un destino evitando colisiones.

---
# 2. Campo de Potencial Artificial
Un **campo de potencial artificial** es una técnica de navegación para robots móviles donde:  
- El **objetivo** genera una fuerza de **atracción**.  
- Los **obstáculos** generan fuerzas de **repulsión**.  
- La suma de estas fuerzas define un “campo” que guía al agente.  

Esto permite que el agente se desplace siguiendo gradientes del campo hasta alcanzar la meta.  
Limitación: puede generar **mínimos locales** donde el robot se queda atrapado (ejemplo: obstáculos en forma de “herradura”).

 Ejemplo: Un dron que vuela hacia una meta mientras esquiva edificios tratados como fuentes de repulsión.

---

# 3. Algoritmo BDI (Belief–Desire–Intention)
El modelo **BDI** describe el comportamiento de un agente en términos de:  
- **Beliefs (Creencias):** lo que el agente sabe del entorno (mapa, sensores, estado actual).  
- **Desires (Deseos):** metas o estados que el agente quiere lograr.  
- **Intentions (Intenciones):** metas seleccionadas y comprometidas a ejecutar, a través de un plan de acciones.  

El flujo es:  
1. El agente **percibe** y actualiza sus *creencias*.  
2. Genera posibles *deseos* en función de sus objetivos.  
3. Selecciona una *intención* concreta.  
4. Ejecuta un plan de acciones hasta cumplirla o hasta que cambien las condiciones.

 Ejemplo: Un robot aspiradora que tiene como **creencia** el mapa de la casa, como **deseo** limpiar todas las habitaciones, y como **intención** limpiar primero la sala porque es la más transitada.

# 3. TERCER PUNTO- SOLUCION A PROBLEMA 
 Entender el algortimo sobre campos de potencia artificial y desarrollar una solucion al problema del agente que se queda dentro de la herradura por medio de algortimo BDI
 # Simulación de Navegación con Campos Potenciales y BDI

Este proyecto muestra cómo un **agente móvil** se desplaza en un entorno con obstáculos usando el método de **campos potenciales**, incorporando un mecanismo de **escape de mínimos locales** inspirado en el modelo **BDI (Belief–Desire–Intention)**.

---

## ¿Qué hace el código?
- El **agente** (rojo) comienza en una posición inicial.  
- Su meta es llegar al **objetivo** (verde).  
- Hay varios **obstáculos** (negros) que forman una especie de herradura.  
- El agente se mueve siguiendo el **gradiente del campo de potencial**:  
  - El **potencial atractivo** lo empuja hacia el objetivo.  
  - El **potencial repulsivo** lo aleja de los obstáculos.  
- Si queda atrapado en un **mínimo local** (cuando apenas se mueve durante varias iteraciones), entra en **modo escape**:  
  - Se mueve aleatoriamente hasta salir de la zona de estancamiento.  
  - Luego vuelve a seguir el gradiente normalmente.  

La simulación se muestra en **dos gráficos**:  
1. El **entorno** con el agente, el objetivo, los obstáculos y la trayectoria.  
2. La evolución de la **energía potencial** a lo largo del tiempo.

---

## Relación con el modelo BDI
El algoritmo se puede entender con la lógica **BDI** de agentes inteligentes:

- **Beliefs (Creencias):**  
  El agente conoce su posición, la ubicación del objetivo y de los obstáculos. También evalúa si está atrapado o no.  

- **Desires (Deseos):**  
  Llegar al objetivo minimizando la energía potencial.  

- **Intentions (Intenciones):**  
  - En **modo normal**, seguir el gradiente del campo potencial.  
  - En **modo escape**, moverse aleatoriamente para salir de trampas locales.  

De esta forma, el agente **razona y actúa** según su percepción del entorno y su estado interno.  

---

##  Visualización
- **Rojo:** posición del agente.  
- **Verde:** objetivo.  
- **Negro:** obstáculos.  
- **Azul:** trayectoria recorrida.  
- **Gráfico lateral:** energía potencial vs iteración.  

El texto en pantalla indica si el agente está en *MODO NORMAL* o en *MODO ESCAPE*.  

---

##  Cómo ejecutar
1. Asegúrate de tener instaladas las librerías:
   ```bash
   pip install numpy matplotlib









