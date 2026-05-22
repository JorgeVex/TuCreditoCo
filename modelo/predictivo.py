import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder


class ModeloPredictivo:
    """Entrena una regresión logística para predecir mora en nuevas solicitudes."""

    FEATURES_NUM = [
        'ratio_endeudamiento',
        'total_income',
        'loan_amount'
    ]
    FEATURES_CAT = [
        'gender',
        'marital_status',
        'contract_type'
    ]

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.modelo = LogisticRegression(max_iter=500, class_weight='balanced', random_state=42)
        self.X_test = None
        self.y_test = None
        self.y_pred = None
        self.y_proba = None
        self.auc: float = 0.0
        self.coefs: pd.DataFrame = pd.DataFrame()

    def entrenar_y_evaluar(self) -> None:
        """Prepara datos, entrena el modelo y evalúa su desempeño."""
        X, y = self._preparar_datos()
        X_train, self.X_test, y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        self.modelo.fit(X_train, y_train)
        self.y_pred  = self.modelo.predict(self.X_test)
        self.y_proba = self.modelo.predict_proba(self.X_test)[:, 1]
        self.auc     = roc_auc_score(self.y_test, self.y_proba)

        self.coefs = pd.DataFrame({
            'Variable': self.FEATURES_NUM + self.FEATURES_CAT,
            'Coeficiente': self.modelo.coef_[0]
        }).sort_values('Coeficiente', ascending=False)

        print(f"\n▸ Tamaño train: {len(X_train):,}  |  test: {len(self.X_test):,}")
        print(f"▸ AUC-ROC: {self.auc:.4f}")
        print("\n▸ Reporte de clasificación:")
        print(classification_report(self.y_test, self.y_pred,
                                    target_names=['Sin mora', 'En mora']))
        print("\n▸ Coeficientes (mayor valor = mayor riesgo de mora):")
        print(self.coefs.to_string(index=False))

    def _preparar_datos(self):
        """Prepara dataset para regresión logística."""
        columnas = self.FEATURES_NUM + self.FEATURES_CAT + ['default']
        df_m = self.df[columnas].copy()

        le = LabelEncoder()
        for col in self.FEATURES_CAT:
            df_m[col] = df_m[col].astype(str)
            df_m[col] = le.fit_transform(df_m[col])

        df_m.replace([np.inf, -np.inf], np.nan, inplace=True)

        print("\n▸ NaN por columna:")
        print(df_m.isna().sum())

        df_m = df_m.dropna(subset=self.FEATURES_NUM)
        print(f"\n▸ Registros válidos: {len(df_m):,}")

        p99 = df_m['ratio_endeudamiento'].quantile(0.99)
        df_m['ratio_endeudamiento'] = df_m['ratio_endeudamiento'].clip(upper=p99)

        X = df_m[self.FEATURES_NUM + self.FEATURES_CAT]
        y = df_m['default']

        return X, y
