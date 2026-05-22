import pandas as pd
import numpy as np


class TransformadorDatos:
    """Aplica transformaciones y crea variables derivadas sobre el DataFrame."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def transformar(self) -> pd.DataFrame:
        """Ejecuta todas las transformaciones necesarias."""
        self._convertir_tipos()
        self._convertir_edad()
        self._calcular_ratio_endeudamiento()
        self._crear_cuartiles()
        self._describir_variables_clave()
        return self.df

    def _convertir_tipos(self) -> None:
        """Convierte únicamente columnas object a numéricas."""
        columnas_numericas = [
            'age',
            'work_age',
            'external_score_1',
            'external_score_2',
            'num_children',
            'age_mobilephone_days',
            'num_petic_bureau_day'
        ]

        for col in columnas_numericas:
            if col not in self.df.columns:
                continue

            self.df[col] = (
                self.df[col]
                .astype(str)
                .str.strip()
                .str.replace(',', '.', regex=False)
            )

            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

    def _convertir_edad(self) -> None:
        """Convierte age y work_age de días negativos a años positivos."""
        self.df['edad_anos'] = (self.df['age'].abs() / 365).round(1)
        self.df['antiguedad_laboral_anos'] = (self.df['work_age'].abs() / 365).round(1)

    def _calcular_ratio_endeudamiento(self) -> None:
        """Calcula ratio crédito/ingreso evitando división por cero."""
        print("\nDEBUG total_income:")
        print(self.df['total_income'].head())

        print("\nDEBUG loan_amount:")
        print(self.df['loan_amount'].head())

        print("\nTipos:")
        print(self.df[['total_income', 'loan_amount']].dtypes)

        self.df['ratio_endeudamiento'] = np.where(
            (self.df['total_income'] > 0) & (self.df['loan_amount'] > 0),
            self.df['loan_amount'] / self.df['total_income'],
            np.nan
        )

        print("\nDEBUG ratio:")
        print(self.df['ratio_endeudamiento'].head())

        print("\nCantidad válidos:")
        print(self.df['ratio_endeudamiento'].notna().sum())

    def _crear_cuartiles(self) -> None:
        """Segmenta el ratio en cuartiles para análisis de riesgo."""
        try:
            cuartiles = pd.qcut(
                self.df['ratio_endeudamiento'],
                q=4,
                duplicates='drop'
            )

            categorias = [
                'Q1 (0-25%)',
                'Q2 (25-50%)',
                'Q3 (50-75%)',
                'Q4 (75-100%)'
            ]

            n_bins = len(cuartiles.cat.categories)

            self.df['cuartil_ratio'] = pd.qcut(
                self.df['ratio_endeudamiento'],
                q=n_bins,
                labels=categorias[:n_bins],
                duplicates='drop'
            )

        except Exception as e:
            print(f"\n⚠ Error creando cuartiles: {e}")
            self.df['cuartil_ratio'] = 'Sin clasificación'

    def _describir_variables_clave(self) -> None:
        """Imprime estadísticas de las variables transformadas."""
        vars_clave = [
            'total_income',
            'loan_amount',
            'ratio_endeudamiento',
            'edad_anos',
            'external_score_1',
            'external_score_2'
        ]

        print("\n▸ Variables transformadas:")

        for v in ['edad_anos', 'antiguedad_laboral_anos', 'ratio_endeudamiento']:
            s = self.df[v]
            print(
                f"  - {v:<28} | "
                f"min: {s.min():.1f} | "
                f"max: {s.max():.1f} | "
                f"media: {s.mean():.1f}"
            )

        print("\n▸ Estadísticas descriptivas (variables clave):")
        print(self.df[vars_clave].describe().round(2).to_string())
