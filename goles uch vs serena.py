# regresion_goles_u_vs_laserena.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

datos = {
    'Tiros': [],
    'Pases_Clave': [],
    'Corners': [],
    'Goles_U': []
}

df = pd.DataFrame(datos)

X = df[['Tiros', 'Pases_Clave', 'Corners']]
y = df['Goles_U']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Resultados de la Evaluación (Goles de U. de Chile vs La Serena)")
print(f"RMSE: {rmse:.2f} (en promedio, las predicciones de goles se desvían en {rmse:.2f})")
print(f"R^2: {r2:.2f} (el {r2:.0%} de la variación en los goles es explicada por las variables)")
