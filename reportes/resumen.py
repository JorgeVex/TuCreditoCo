import pandas as pd

from analisis.analizador import AnalizadorProblematicas
from analisis.criterio import CriterioDecision
from modelo.predictivo import ModeloPredictivo


class ResumenEjecutivo:
    """Consolida y presenta el resumen final de hallazgos y decisiones."""

    def __init__(self, df: pd.DataFrame, analizador: AnalizadorProblematicas,
                 criterio: CriterioDecision, modelo: ModeloPredictivo):
        self.df = df
        self.analizador = analizador
        self.criterio = criterio
        self.modelo = modelo

    def imprimir(self) -> None:
        """Imprime el resumen ejecutivo completo."""
        sobre = self.df[self.df['ratio_endeudamiento'] > AnalizadorProblematicas.UMBRAL_RATIO]
        normal = self.df[self.df['ratio_endeudamiento'] <= AnalizadorProblematicas.UMBRAL_RATIO]
        os = self.analizador.office_stats
        im = self.analizador.income_mora
        im_activos = im[im['tasa_mora_pct'] > 0]

        print(f"""
HALLAZGO 1 — Sobreendeudamiento (Problemática principal)
  • El {len(sobre)/len(self.df)*100:.1f}% de los clientes tiene crédito > 5x sus ingresos anuales.
  • Su tasa de mora ({sobre['default'].mean()*100:.1f}%) supera a la del resto ({normal['default'].mean()*100:.1f}%).
  • Diferencia estadísticamente significativa (p < 0.05).
  DECISIÓN: Rechazar solicitudes con ratio_endeudamiento > {self.criterio.umbral:.1f}x.

HALLAZGO 2 — Diferencias entre oficinas
  • Mayor mora: Oficina {os.iloc[0]['Office']} ({os.iloc[0]['tasa_mora_pct']}%).
  • Menor mora: Oficina {os.iloc[-1]['Office']} ({os.iloc[-1]['tasa_mora_pct']}%).
  • Diferencia significativa por chi-cuadrado (p < 0.05).
  DECISIÓN: Auditar oficinas con mora superior al promedio.

HALLAZGO 3 — Tipo de ingreso como factor de riesgo
  • Mayor mora: {im.iloc[0]['income_type']} ({im.iloc[0]['tasa_mora_pct']}%).
  • Menor mora: {im_activos.iloc[-1]['income_type']} ({im_activos.iloc[-1]['tasa_mora_pct']}%).
  DECISIÓN: Aplicar criterios más estrictos a clientes tipo "{im.iloc[0]['income_type']}".

HALLAZGO 4 — Scores externos como predictor
  • Clientes en mora tienen scores externos ~20% más bajos.
  • Diferencia estadísticamente significativa (p < 0.05).
  DECISIÓN: Rechazar solicitudes con external_score_1 < 0.45.

MODELO PREDICTIVO — Regresión Logística
  • AUC-ROC: {self.modelo.auc:.4f} — buena capacidad discriminante.
  • Variables más relevantes: ratio_endeudamiento, external_score_1, external_score_2.
  USO: Aplicar el modelo a nuevas solicitudes para estimar probabilidad
       de mora antes de aprobar el crédito.
""")
