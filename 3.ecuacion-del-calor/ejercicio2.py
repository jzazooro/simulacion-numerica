import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parámetros dados
b = 5
d = 10
N = 40
M = 400
alpha = 0.3

# Calculando Δx y Δt
dx = b / N
dt = d / M

# Creando la malla
x = np.linspace(0, b, N+1)
t = np.linspace(0, d, M+1)

# Inicializando la solución u con ceros
u = np.zeros((N+1, M+1))

# Condición inicial
u[:, 0] = [1 if 0 < xi < b/2 else 0 for xi in x]

# Coeficiente de difusión térmica
sigma = alpha * dt / dx**2

# Aplicando el método de diferencias finitas
for j in range(0, M):
    for i in range(1, N):
        u[i, j+1] = u[i, j] + sigma * (u[i-1, j] - 2*u[i, j] + u[i+1, j])

# Preparando para el gráfico 3D
T, X = np.meshgrid(t, x)

# Configurando la figura y el eje para 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(T, X, u, cmap='viridis')

# Etiquetas y título
ax.set_xlabel('Tiempo')
ax.set_ylabel('Posición en la barra')
ax.set_zlabel('Temperatura')
ax.set_title('Evolución de la temperatura en la barra en 3D')
fig.colorbar(surf, shrink=0.5, aspect=5, label='Temperatura')

plt.show()


alphamax=dx/np.sqrt(2*dt)
print("el valo mazimo de alpha es: ", alphamax)