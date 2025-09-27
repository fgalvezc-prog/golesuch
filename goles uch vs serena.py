import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

#Datos sacados de sofascore.com

datos = {
    'Posesion balón': [43, 67, 53, 59, 68, 60, 39, 40, 58, 63, 44, 63, 60, 73, 54, 67, 65, 56, 61, 66, 63, 66, 43, 44, 65, 59, 55, 70, 66, 53, 54, 69, 34, 56, 59, 50, ],
    'Tiros': [6, 36, 13, 14, 18, 23, 6, 11, 20, 11, 17, 15, 13, 23, 16, 22, 17, 16, 9, 24, 11, 23, 12, 10, 21, 10, 11, 17, 25, 24, 13, 19, 12, 16, 13, 14, ],
    'Tiros al arco': [2, 14, 3, 4, 3, 3, 2, 6, 6, 2, 6, 3, 5, 9, 8, 8, 1, 7, 5, 5, 3, 9, 4, 4, 9, 5, 5, 8, 4, 8, 4, 5, 2, 9, 2, 6, ],
    'Atajadas del rival': [1, 9, 2, 3, 2, 2, 1, 4, 3, 1, 3, 2, 5, 5, 3, 8, 1, 1, 4, 3, 2, 6, 2, 2, 4, 4, 4, 4, 4, 4, 3, 4, 2, 6, 2, 3, ],
    'Oportunidades clave': [2, 7, 3, 0, 1, 2, 1, 3, 5, 2, 2, 2, 0, 3, 5, 3, 1, 6, 4, 0, 2, 6, 2, 2, 2, 2, 2, 3, 1, 5, 0, 4, 2, 2, 0, 4, ],
    'Corners': [1, 15, 7, 11, 4, 5, 2, 4, 3, 6, 5, 5, 7, 8, 8, 5, 5, 3, 5, 8, 4, 7, 3, 1, 1, 3, 4, 3, 11, 5, 6, 8, 7, 4, 6, 3, ],
    'Goles_U': [1, 5, 1, 1, 1, 0, 1, 2, 3, 1, 3, 1, 0, 4, 5, 0, 0, 6, 1, 2, 0, 3, 2, 2, 5, 2, 1, 4, 0, 4, 1, 1, 0, 3, 0, 2, ]
}

df = pd.DataFrame(datos)

variables = ['Posesion balón', 'Tiros', 'Tiros al arco', 'Atajadas del rival', 'Oportunidades clave', 'Corners']
X = df[variables]
y = df['Goles_U']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

prom_global = df[variables].mean().to_frame().T
prom_last5  = df[variables].tail(5).mean().to_frame().T
prom_last3  = df[variables].tail(3).mean().to_frame().T

pred_global = float(model.predict(prom_global)[0])
pred_last5  = float(model.predict(prom_last5)[0])
pred_last3  = float(model.predict(prom_last3)[0])

valores = np.array([pred_global, pred_last5, pred_last3])
pred_recomendada = float(np.median(valores))
pred_recomendada_red = max(0, round(pred_recomendada))

print(f"Resultados de la Evaluación (Goles de U. de Chile vs La Serena)")
print(f"RMSE: {rmse:.2f} (en promedio, las predicciones de goles se desvían en {rmse:.2f})")
print(f"R^2: {r2:.2f} (el {r2:.0%} de la variación en los goles es explicada por las variables)")

print("\nPronóstico U. de Chile vs La Serena")
print(f"  Global de partidos jugados en el año  : {pred_global:.2f} goles")
print(f"  Últimos 5 partidos jugados : {pred_last5:.2f} goles")
print(f"  Últimos 3 partidos jugados : {pred_last3:.2f} goles")
print(f"\n  ► Pronóstico: La U de Chile hará {pred_recomendada:.2f}  → {pred_recomendada_red} gol(es)")
