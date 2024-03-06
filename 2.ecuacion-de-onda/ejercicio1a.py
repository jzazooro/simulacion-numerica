import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ecuacion de onda

# Parámetros dados
b = 5
d = 10
n = 40
m = 400
v = 0.5

# Calculando Δx, Δt y r
Δx = b / n
Δt = d / m
r = v * Δt / Δx

# Inicializando la matriz u para almacenar las posiciones de la cuerda
u = np.zeros((n+1, m+1))

# Condiciones iniciales
x = np.linspace(0, b, n+1)
u[:, 0] = x * (b - x)  # f(x) = x(b-x)
u[:, 1] = u[:, 0]  # Aproximando que la primera derivada temporal es 0

# Aplicando el método de diferencias finitas
for j in range(1, m):
    for i in range(1, n):
        u[i, j+1] = (2 - 2*r**2)*u[i, j] + r**2*(u[i+1, j] + u[i-1, j]) - u[i, j-1]

# Graficando la cuerda en 3D
X, Y = np.meshgrid(x, np.linspace(0, d, m+1))

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Graficando la superficie
ax.plot_surface(X, Y, u.T, cmap='viridis')

ax.set_title('Evolución de la cuerda en el tiempo en 3D')
ax.set_xlabel('Posición a lo largo de la cuerda (m)')
ax.set_ylabel('Tiempo (s)')
ax.set_zlabel('Desplazamiento de la cuerda (m)')
plt.show()

# Hallar la velocidad maxima de propagación de la onda
v_max = Δx / Δt

print('La velocidad máxima de propagación de la onda es:', v_max)