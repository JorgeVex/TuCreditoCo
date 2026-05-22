import pandas as pd
from scipy import stats


class AnalizadorProblematicas:
    """Identifica, cuantifica y valida estadísticamente las problemáticas."""

    UMBRAL_RATIO = 5       # Veces el ingreso anual
    UMBRAL_SCORE = 0.45    # Score externo mínimo aceptable

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.mora_total = df['default'].mean()
        self.office_stats: pd.DataFrame = pd.DataFrame()
        self.income_mora: pd.DataFrame = pd.DataFrame()

    def ejecutar_analisis(self) -> None:
        """Corre todas las problemáticas en secuencia."""
        print(f"\n▸ Tasa de mora global: {self.mora_total*100:.2f}%  "
              f"({self.df['default'].sum():,} clientes de {len(self.df):,})")
        self._problematica_sobreendeudamiento()
        self._problematica_oficinas()
        self._problematica_tipo_ingreso()
        self._problematica_perfil_demografico()
        self._problematica_scores_externos()

    def _problematica_sobreendeudamiento(self) -> None:
        """P1: Clientes con créditos muy superiores a su capacidad de pago."""
        print("\n── PROBLEMÁTICA 1: Sobreendeudamiento ──────────────────────")
        sobre = self.df[self.df['ratio_endeudamiento'] > self.UMBRAL_RATIO]
        normal = self.df[self.df['ratio_endeudamiento'] <= self.UMBRAL_RATIO]
        print(f"  Clientes con ratio > {self.UMBRAL_RATIO}x ingresos: {len(sobre):,} "
              f"({len(sobre)/len(self.df)*100:.1f}%)")
        print(f"  Mora en sobreendeudados:      {sobre['default'].mean()*100:.2f}%")
        print(f"  Mora en endeudamiento normal: {normal['default'].mean()*100:.2f}%")

        g0 = self.df[self.df['default'] == 0]['ratio_endeudamiento']
        g1 = self.df[self.df['default'] == 1]['ratio_endeudamiento']
        t, p = stats.ttest_ind(g0, g1, equal_var=False)
        print(f"\n  Test t (ratio: mora vs sin mora):")
        print(f"    Media sin mora: {g0.mean():.3f}  |  Media con mora: {g1.mean():.3f}")
        print(f"    t = {t:.4f}  |  p-value = {p:.4e}")
        print(f"    → {'Diferencia ESTADÍSTICAMENTE SIGNIFICATIVA (p<0.05)' if p < 0.05 else 'Sin diferencia'}")

    def _problematica_oficinas(self) -> None:
        """P2: Diferencias en tasa de mora entre oficinas."""
        print("\n── PROBLEMÁTICA 2: Diferencias entre oficinas ───────────────")
        self.office_stats = self.df.groupby('Office').agg(
            total=('default', 'count'),
            mora=('default', 'sum'),
            ratio_medio=('ratio_endeudamiento', 'mean')
        ).reset_index()
        self.office_stats['tasa_mora_pct'] = (
            self.office_stats['mora'] / self.office_stats['total'] * 100
        ).round(2)
        self.office_stats = self.office_stats.sort_values('tasa_mora_pct', ascending=False)
        print(self.office_stats.to_string(index=False))

        tabla = pd.crosstab(self.df['Office'], self.df['default'])
        chi2, p_chi, dof, _ = stats.chi2_contingency(tabla)
        print(f"\n  Chi-cuadrado (mora vs oficina):")
        print(f"    chi2 = {chi2:.2f}  |  p-value = {p_chi:.4e}  |  gl = {dof}")
        print(f"    → {'Diferencia SIGNIFICATIVA entre oficinas (p<0.05)' if p_chi < 0.05 else 'Sin diferencia'}")

    def _problematica_tipo_ingreso(self) -> None:
        """P3: Mora según el tipo de ingreso del solicitante."""
        print("\n── PROBLEMÁTICA 3: Mora por tipo de ingreso ─────────────────")
        self.income_mora = self.df.groupby('income_type').agg(
            total=('default', 'count'),
            mora=('default', 'sum')
        ).reset_index()
        self.income_mora['tasa_mora_pct'] = (
            self.income_mora['mora'] / self.income_mora['total'] * 100
        ).round(2)
        self.income_mora = self.income_mora.sort_values('tasa_mora_pct', ascending=False)
        print(self.income_mora.to_string(index=False))

    def _problematica_perfil_demografico(self) -> None:
        """P4: Mora según estado civil y género."""
        print("\n── PROBLEMÁTICA 4: Mora por estado civil y género ───────────")
        print("  Por estado civil:")
        print(self.df.groupby('marital_status')['default']
              .mean().round(4).sort_values(ascending=False).to_string())
        print("\n  Por género:")
        print(self.df.groupby('gender')['default']
              .mean().round(4).sort_values(ascending=False).to_string())

    def _problematica_scores_externos(self) -> None:
        """P5: Scores externos como señal de riesgo crediticio."""
        print("\n── PROBLEMÁTICA 5: Scores externos vs mora ──────────────────")
        print(self.df.groupby('default')[['external_score_1', 'external_score_2']]
              .mean().round(3).to_string())
        t, p = stats.ttest_ind(
            self.df[self.df['default'] == 0]['external_score_1'],
            self.df[self.df['default'] == 1]['external_score_1'],
            equal_var=False
        )
        print(f"\n  Test t (external_score_1): t = {t:.4f}  |  p-value = {p:.4e}")
        print(f"  → {'Diferencia SIGNIFICATIVA (p<0.05)' if p < 0.05 else 'Sin diferencia'}")
