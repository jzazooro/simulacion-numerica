import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# ecuacion de onda

# Parámetros dados
b = 6
d = 24
n = 600
m = 2400
v = 0.5 

# Calculando Δx, Δt y r
Δx = b / n
Δt = d / m
r = v * Δt / Δx

# Inicializando la matriz u para almacenar las posiciones de la cuerda
w = np.zeros((m + 1, n + 1))

# Condiciones iniciales
def f(x):
    return 0

def g(x):
    return 0

# Condiciones de contorno
for i in range(1, n):
    w[0][i] = f(Δx * i)
    w[1][i] = w[0][i] + Δt * g(Δx * i)

for j in range(1, m):
    w[j][0] = np.sin(Δt*j)
    w[j][n] = 0

# Aplicando el método de diferencias finitas
for j in range(1, m):
    for i in range(1, n):
        w[j+1][i] = 2 * (1 - r*2) * w[j][i] - w[j-1][i] + r*2 * (w[j][i+1] + w[j][i-1])

# Graficando la cuerda en 3D
X = np.linspace(0, b, n+1)
Y = np.linspace(0, d, m+1)
X, Y = np.meshgrid(X, Y)

# Graficando la superficie
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, w, cmap='viridis')
ax.set_xlabel('Posición a lo largo de la cuerda (m)')
ax.set_ylabel('Tiempo (s)')
ax.set_zlabel('Desplazamiento de la cuerda (m)')

plt.show()