# =============================================================================
# ACTIVIDAD 2 - CIENCIA DE DATOS PARA LA TOMA DE DECISIONES ESTRATÉGICAS
# Empresa: TuCréditoCo | Versión: Programación Orientada a Objetos (POO)
# =============================================================================
# INSTRUCCIONES DE USO:
#   1. Coloca este script dentro de tu carpeta "tucreditoco"
#   2. Asegúrate de que el dataset esté en la misma carpeta que main.py
#   3. Ejecuta: python main.py
#
# LIBRERÍAS NECESARIAS:
#   pip install pandas numpy matplotlib seaborn scipy scikit-learn
# =============================================================================

import sys
import os
import warnings

warnings.filterwarnings('ignore')

# Permite que Python encuentre los paquetes internos del proyecto
sys.path.insert(0, os.path.dirname(__file__))

from datos.cargador import CargadorDatos
from datos.transformador import TransformadorDatos
from analisis.analizador import AnalizadorProblematicas
from analisis.criterio import CriterioDecision
from modelo.predictivo import ModeloPredictivo
from visualizacion.visualizador import Visualizador
from reportes.resumen import ResumenEjecutivo


class Pipeline:
    """Orquesta la ejecución completa del análisis de principio a fin."""

    SEPARADOR = "=" * 65

    def __init__(self, ruta_csv: str):
        self.ruta_csv = ruta_csv

    def ejecutar(self) -> None:
        """Corre el pipeline completo en secuencia."""
        self._encabezado("ACTIVIDAD 2 - ANÁLISIS DE DATOS TUCRÉDITOCO")

        # Paso 1: Cargar datos
        self._encabezado("PASO 1: FAMILIARIZACIÓN CON LOS DATOS")
        cargador = CargadorDatos(self.ruta_csv)
        df_raw = cargador.cargar()
        cargador.resumen_estructura()

        # Paso 2: Transformar datos
        transformador = TransformadorDatos(df_raw)
        df = transformador.transformar()

        # Paso 3: Analizar problemáticas
        self._encabezado("PASO 2: IDENTIFICACIÓN DE PROBLEMÁTICAS")
        analizador = AnalizadorProblematicas(df)
        analizador.ejecutar_analisis()

        # Paso 4: Criterio de decisión
        self._encabezado("PASO 3: CRITERIO DE DECISIÓN")
        criterio = CriterioDecision(df)
        criterio.calcular()

        # Paso 5: Modelo predictivo
        self._encabezado("PASO 4: MODELO PREDICTIVO - REGRESIÓN LOGÍSTICA")
        modelo = ModeloPredictivo(df)
        modelo.entrenar_y_evaluar()

        # Paso 6: Visualizaciones
        self._encabezado("PASO 5: GENERANDO VISUALIZACIONES...")
        visualizador = Visualizador(df, analizador, criterio, modelo)
        visualizador.generar("graficas_tucreditoco.png")

        # Paso 7: Resumen ejecutivo
        self._encabezado("RESUMEN EJECUTIVO — HALLAZGOS Y DECISIONES")
        resumen = ResumenEjecutivo(df, analizador, criterio, modelo)
        resumen.imprimir()

        print(self.SEPARADOR)
        print("  Análisis finalizado.")
        print(self.SEPARADOR)

    def _encabezado(self, titulo: str) -> None:
        print(f"\n{self.SEPARADOR}")
        print(f"  {titulo}")
        print(self.SEPARADOR)


# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================
if __name__ == "__main__":
    pipeline = Pipeline(ruta_csv="solicitud_creditos_info.csv")
    pipeline.ejecutar()
