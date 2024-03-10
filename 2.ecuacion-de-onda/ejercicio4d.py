# d: queremos formar ondas no estacionarias (f=g=0). ¿como lo harias?

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Inicialización de parámetros y condiciones iniciales
b = 6.0  # Longitud de la cuerda
d = 24.0  # Duración de la simulación
n = 600  # Pasos espaciales
m = 2400  # Pasos temporales
v = 0.5  # Velocidad de la onda

Δx = b / n
Δt = d / m
r = v * Δt / Δx

# Matriz para las posiciones de la cuerda
u = np.zeros((n+1, m+1))
x = np.linspace(0, b, n+1)

# Condición inicial f(x)
u[:, 0] = np.where((x >= 1) & (x <= 3), 1, 0)

# Condiciones de contorno fijas en ambos extremos
for j in range(0, m):
    for i in range(1, n):
        u[i, j+1] = (2 - 2*r**2)*u[i, j] + r**2*(u[i+1, j] + u[i-1, j]) - u[i, j-1]
    u[0, j+1] = 0  # Extremo izquierdo fijo
    u[n, j+1] = 0  # Extremo derecho fijo

# Configuración para la visualización 3D
X, Y = np.meshgrid(x, np.linspace(0, d, m+1))

fig = plt.figure(figsize=(14, 9))
ax = fig.add_subplot(111, projection='3d')

# Graficando la superficie
ax.plot_surface(X, Y, u.T, cmap='viridis')

ax.set_title('Evolución de la cuerda con ondas no estacionarias y extremos fijos')
ax.set_xlabel('Posición a lo largo de la cuerda (m)')
ax.set_ylabel('Tiempo (s)')
ax.set_zlabel('Desplazamiento de la cuerda (m)')

plt.show()