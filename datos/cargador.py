import pandas as pd


class CargadorDatos:
    """Carga el dataset desde un archivo CSV y valida su estructura básica."""

    def __init__(self, ruta: str):
        self.ruta = ruta
        self.df: pd.DataFrame = pd.DataFrame()

    def cargar(self) -> pd.DataFrame:
        """Lee el CSV y retorna el DataFrame."""
        self.df = pd.read_csv(self.ruta, sep=';')

        # Eliminar columnas basura tipo "Unnamed: 33"
        self.df = self.df.loc[
            :,
            ~self.df.columns.str.contains('^Unnamed')
        ]

        print(f"✔ Dataset cargado: {self.ruta}")
        return self.df

    def resumen_estructura(self) -> None:
        """Imprime dimensiones, tipos y valores faltantes."""
        print(f"\n▸ Registros (filas): {self.df.shape[0]:,}")
        print(f"▸ Variables (columnas): {self.df.shape[1]}")

        tipos = self.df.dtypes.reset_index()
        tipos.columns = ['Variable', 'Tipo']
        tipos['Clasificación'] = tipos['Tipo'].apply(
            lambda t: 'Numérica continua' if t == 'float64'
            else ('Numérica entera' if t == 'int64' else 'Categórica / Texto')
        )
        print("\n▸ Tipos de datos por variable:")
        print(tipos.to_string(index=False))

        print("\n▸ Revisión de formatos:")
        print("  - 'age' y 'work_age': float64 representando días negativos → convertir a años.")
        print("  - 'default': int64 binaria (0/1) → tratar como categórica en gráficas.")
        print("  - 'Office' y 'Employee': int64 pero son identificadores, no métricas.")

        nulos = self.df.isnull().sum()
        nulos_pct = (nulos / len(self.df) * 100).round(2)
        resumen = pd.DataFrame({'Nulos': nulos, '% del total': nulos_pct})
        resumen = resumen[resumen['Nulos'] > 0]
        print("\n▸ Valores faltantes:")
        if resumen.empty:
            print("  ✔ No se detectaron valores faltantes.")
        else:
            print(resumen.to_string())
