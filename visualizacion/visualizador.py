import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

from analisis.analizador import AnalizadorProblematicas
from analisis.criterio import CriterioDecision
from modelo.predictivo import ModeloPredictivo


class Visualizador:
    """Genera el panel de 6 gráficas del análisis de riesgo crediticio."""

    COLORES = ['#2E75B6', '#E74C3C', '#27AE60', '#F39C12', '#8E44AD', '#16A085']

    def __init__(self, df: pd.DataFrame, analizador: AnalizadorProblematicas,
                 criterio: CriterioDecision, modelo: ModeloPredictivo):
        self.df = df
        self.analizador = analizador
        self.criterio = criterio
        self.modelo = modelo
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'axes.spines.top': False,
            'axes.spines.right': False,
            'figure.dpi': 120
        })

    def generar(self, ruta_salida: str = "graficas_tucreditoco.png") -> None:
        """Crea y guarda el panel completo de visualizaciones."""
        fig, axes = plt.subplots(3, 2, figsize=(18, 14))
        fig.suptitle(
            "TuCréditoCo — Análisis de Riesgo Crediticio",
            fontsize=22, fontweight='bold', color='#1F5C99', y=0.98
        )

        self._grafica_distribucion_mora(axes[0, 0])
        self._grafica_mora_por_cuartil(axes[0, 1])
        self._grafica_mora_por_oficina(axes[1, 0])
        self._grafica_distribucion_ratio(axes[1, 1])
        self._grafica_scores_externos(axes[2, 0])
        self._grafica_curva_roc(axes[2, 1])

        plt.subplots_adjust(
            hspace=0.40, wspace=0.25,
            top=0.92, bottom=0.06,
            left=0.06, right=0.97
        )

        plt.savefig(ruta_salida, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"\n✔ Gráficas guardadas en: {ruta_salida}")

    def _grafica_distribucion_mora(self, ax) -> None:
        sizes  = [self.df['default'].value_counts()[0],
                  self.df['default'].value_counts()[1]]
        labels = [f'Sin mora\n({sizes[0]/sum(sizes)*100:.1f}%)',
                  f'En mora\n({sizes[1]/sum(sizes)*100:.1f}%)']
        wedges, _, autotexts = ax.pie(
            sizes, labels=labels, colors=[self.COLORES[0], self.COLORES[1]],
            autopct='%1.1f%%', startangle=90,
            wedgeprops={'edgecolor': 'white', 'linewidth': 2}
        )
        for at in autotexts:
            at.set_fontsize(11); at.set_fontweight('bold'); at.set_color('white')
        ax.set_title("Distribución de mora en el portafolio",
                     fontweight='bold', color='#1F5C99')

    def _grafica_mora_por_cuartil(self, ax) -> None:
        cm = self.criterio.cuartil_mora
        colores = ['#27AE60', '#F39C12', '#E67E22', '#E74C3C']
        bars = ax.bar(cm['cuartil_ratio'].astype(str), cm['tasa_mora_pct'],
                      color=colores, edgecolor='white', linewidth=1.5)
        for bar, val in zip(bars, cm['tasa_mora_pct']):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{val:.1f}%', ha='center', va='bottom',
                    fontweight='bold', fontsize=10)
        ax.set_title("Tasa de mora por cuartil de endeudamiento",
                     fontweight='bold', color='#1F5C99')
        ax.set_xlabel("Cuartil de ratio crédito/ingreso")
        ax.set_ylabel("Tasa de mora (%)")
        ax.set_ylim(0, cm['tasa_mora_pct'].max() * 1.25)

    def _grafica_mora_por_oficina(self, ax) -> None:
        os = self.analizador.office_stats
        colores = ['#E74C3C' if t > self.analizador.mora_total * 100
                   else '#2E75B6' for t in os['tasa_mora_pct']]
        ax.bar(os['Office'].astype(str), os['tasa_mora_pct'],
               color=colores, edgecolor='white', linewidth=1)
        ax.axhline(self.analizador.mora_total * 100, color='#F39C12',
                   linestyle='--', linewidth=1.5,
                   label=f'Promedio: {self.analizador.mora_total*100:.1f}%')
        ax.set_title("Tasa de mora por oficina", fontweight='bold', color='#1F5C99')
        ax.set_xlabel("Oficina"); ax.set_ylabel("Tasa de mora (%)")
        ax.legend(fontsize=9); ax.tick_params(axis='x', rotation=90)

    def _grafica_distribucion_ratio(self, ax) -> None:
        ratio_clip = self.df['ratio_endeudamiento'].clip(
            upper=self.df['ratio_endeudamiento'].quantile(0.99)
        )
        ax.hist(ratio_clip[self.df['default'] == 0], bins=60, alpha=0.6,
                color=self.COLORES[0], label='Sin mora', density=True)
        ax.hist(ratio_clip[self.df['default'] == 1], bins=60, alpha=0.6,
                color=self.COLORES[1], label='En mora', density=True)
        ax.axvline(self.criterio.umbral, color='#F39C12', linestyle='--',
                   linewidth=2, label=f'Umbral p75: {self.criterio.umbral:.1f}x')
        ax.set_title("Distribución del ratio crédito/ingreso por grupo",
                     fontweight='bold', color='#1F5C99')
        ax.set_xlabel("Ratio endeudamiento (limitado a 20x)")
        ax.set_ylabel("Densidad"); ax.legend(fontsize=9)

    def _grafica_scores_externos(self, ax) -> None:
        scores = self.df.groupby('default')[['external_score_1', 'external_score_2']].mean()
        scores.index = ['Sin mora', 'En mora']
        scores.plot(kind='bar', ax=ax, color=[self.COLORES[0], self.COLORES[4]],
                    edgecolor='white', width=0.6)
        ax.set_title("Score externo promedio por grupo",
                     fontweight='bold', color='#1F5C99')
        ax.set_xlabel(""); ax.set_ylabel("Score promedio")
        ax.tick_params(axis='x', rotation=0)
        ax.legend(['External Score 1', 'External Score 2'], fontsize=9)
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', fontsize=9, padding=2)
        ax.set_ylim(0, 0.7)

    def _grafica_curva_roc(self, ax) -> None:
        fpr, tpr, _ = roc_curve(self.modelo.y_test, self.modelo.y_proba)
        ax.plot(fpr, tpr, color=self.COLORES[0], linewidth=2,
                label=f'ROC (AUC = {self.modelo.auc:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Clasificador aleatorio')
        ax.fill_between(fpr, tpr, alpha=0.1, color=self.COLORES[0])
        ax.set_title("Curva ROC — Regresión Logística",
                     fontweight='bold', color='#1F5C99')
        ax.set_xlabel("Tasa de Falsos Positivos")
        ax.set_ylabel("Tasa de Verdaderos Positivos")
        ax.legend(fontsize=10)
