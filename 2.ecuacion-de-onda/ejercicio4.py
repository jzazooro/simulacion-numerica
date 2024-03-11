import numpy as np
import matplotlib.pyplot as plt
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
u = np.zeros((n+1, m+1))
x = np.linspace(0, b, n+1)

# Condiciones iniciales
u[:, 0] = np.where((x >= 1) & (x <= 3), 1, 0)
u[:, 1] = u[:, 0]  # Aproximando que la primera derivada temporal es 0

# Aplicando el método de diferencias finitas
for j in range(1, m):
    for i in range(1, n):
        u[i, j+1] = (2 - 2*r**2)*u[i, j] + r**2*(u[i+1, j] + u[i-1, j]) - u[i, j-1]

# Graficando la cuerda en 3D
X, Y = np.meshgrid(x, np.linspace(0, d, m+1))
fig = plt.figure(figsize=(14, 9))
ax = fig.add_subplot(111, projection='3d')

# Graficando la superficie
ax.plot_surface(X, Y, u.T, cmap='viridis')

ax.set_title('Evolución de la cuerda en el tiempo con condición inicial modificada')
ax.set_xlabel('Posición a lo largo de la cuerda')
ax.set_ylabel('Tiempo')
ax.set_zlabel('Desplazamiento de la cuerda')
plt.show()
