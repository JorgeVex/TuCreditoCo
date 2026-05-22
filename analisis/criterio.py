import pandas as pd


class CriterioDecision:
    """Define el umbral de rechazo de solicitudes con base en los datos."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.umbral: float = 0.0
        self.cuartil_mora: pd.DataFrame = pd.DataFrame()

    def calcular(self) -> None:
        """Calcula el umbral p75 y evalúa su impacto en la mora."""
        self.cuartil_mora = self.df.groupby('cuartil_ratio', observed=True).agg(
            total=('default', 'count'),
            mora=('default', 'sum'),
            ratio_min=('ratio_endeudamiento', 'min'),
            ratio_max=('ratio_endeudamiento', 'max')
        ).reset_index()
        self.cuartil_mora['tasa_mora_pct'] = (
            self.cuartil_mora['mora'] / self.cuartil_mora['total'] * 100
        ).round(2)

        ratio_limpio = self.df['ratio_endeudamiento'].clip(
            upper=self.df['ratio_endeudamiento'].quantile(0.99)
        )
        self.umbral = ratio_limpio.quantile(0.75)

        mora_bajo = self.df[self.df['ratio_endeudamiento'] <= self.umbral]['default'].mean()
        mora_sobre = self.df[self.df['ratio_endeudamiento'] > self.umbral]['default'].mean()

        print("\n▸ Tasa de mora por cuartil de ratio de endeudamiento:")
        print(self.cuartil_mora[['cuartil_ratio', 'ratio_min', 'ratio_max',
                                  'total', 'mora', 'tasa_mora_pct']].to_string(index=False))
        print(f"\n▸ Umbral propuesto (percentil 75): ratio ≤ {self.umbral:.2f}x ingresos")
        print(f"  Mora con ratio ≤ {self.umbral:.2f}: {mora_bajo*100:.2f}%")
        print(f"  Mora con ratio >  {self.umbral:.2f}: {mora_sobre*100:.2f}%")
        print(f"\n  CRITERIO DE DECISIÓN:")
        print(f"  → Rechazar solicitudes con ratio_endeudamiento > {self.umbral:.2f}x")
        print(f"    Y/o con external_score_1 < 0.45 o external_score_2 < 0.40")
