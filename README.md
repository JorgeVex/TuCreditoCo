# TuCreditoCo – Proyecto de Ciencia de Datos

## Descripción del proyecto

Este proyecto desarrolla un proceso completo de ciencia de datos aplicado a una problemática financiera relacionada con el sobreendeudamiento y el riesgo de mora en clientes de una entidad crediticia ficticia llamada **TuCreditoCo**.

El objetivo principal consiste en analizar el comportamiento financiero de los clientes, identificar patrones asociados al incumplimiento y construir un modelo predictivo que apoye la toma de decisiones sobre aprobación de créditos.

El proyecto incluye:

* Limpieza y transformación de datos.
* Análisis exploratorio.
* Visualización de métricas financieras.
* Validaciones estadísticas.
* Construcción de un modelo predictivo.
* Generación de reportes automatizados.

---

# Objetivo

Identificar la relación entre el nivel de endeudamiento y el riesgo de mora de los clientes, utilizando técnicas de análisis de datos y machine learning para apoyar procesos de evaluación crediticia.

---

# Variables principales

| Variable            | Descripción                       |
| ------------------- | --------------------------------- |
| loan_amount         | Monto del préstamo                |
| total_income        | Ingresos totales del cliente      |
| ratio_endeudamiento | Relación entre crédito e ingresos |
| default             | Incumplimiento de pago            |
| external_score_1    | Puntaje externo 1                 |
| external_score_2    | Puntaje externo 2                 |

---

# Estructura del proyecto

```bash
TuCreditoCo/
│
├── analisis/
│   ├── analisis_crediticio.py
│   ├── estadisticas.py
│
├── datos/
│   ├── cargador.py
│   ├── transformaciones.py
│
├── modelo/
│   ├── modelo_riesgo.py
│
├── reportes/
│   ├── reporte.py
│
├── visualizacion/
│   ├── graficas.py
│
├── main.py
├── requirements.txt
└── README.md
```

---

# Tecnologías utilizadas

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* SciPy

---

# Metodología aplicada

El proyecto sigue un flujo de trabajo de ciencia de datos compuesto por:

1. Comprensión del problema.
2. Exploración y limpieza de datos.
3. Ingeniería de variables.
4. Análisis exploratorio.
5. Validación estadística.
6. Construcción del modelo predictivo.
7. Evaluación de resultados.
8. Generación de conclusiones y reportes.

---

# Hallazgos principales

* Los clientes con mayores niveles de endeudamiento presentan una mayor probabilidad de mora.
* El ratio crédito/ingreso es una variable crítica para evaluar riesgo financiero.
* Los puntajes externos ayudan a complementar la evaluación crediticia.
* El modelo predictivo permite estimar probabilidades de incumplimiento y apoyar la toma de decisiones.

---

# Modelo predictivo

Se implementó un modelo de **Regresión Logística** para predecir la probabilidad de incumplimiento de un cliente.

Las métricas utilizadas para evaluar el desempeño incluyen:

* Accuracy
* Precision
* Recall
* Matriz de confusión
* Curva ROC
* AUC

---

# Instalación

Clonar el repositorio:

```bash
git clone https://github.com/JorgeVex/TuCreditoCo.git
```

Ingresar al proyecto:

```bash
cd TuCreditoCo
```

Instalar dependencias:

```bash
pip install -r requirements.txt
```

---

# Ejecución

Para ejecutar el proyecto:

```bash
python main.py
```

---

# Resultados esperados

El sistema permite:

* Analizar perfiles financieros de clientes.
* Detectar patrones de riesgo.
* Visualizar métricas relevantes.
* Generar reportes automáticos.
* Estimar probabilidades de mora mediante machine learning.

---

# Consideraciones

Este proyecto fue desarrollado con fines académicos para aplicar técnicas de ciencia de datos en un contexto financiero.

Los resultados obtenidos dependen de la calidad y características del dataset utilizado.

---

# Integrantes

* Jorge Hernán Velasco Gómez
* Kenny Felipe Mape Silva
* Yuleidy Cruz Valbuena

---

# Repositorio

Repositorio oficial del proyecto:

[https://github.com/JorgeVex/TuCreditoCo](https://github.com/JorgeVex/TuCreditoCo)
