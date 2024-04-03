import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ecuacion de calor

# Parámetros dados
b = 5 # longitud de la barra
d = 10 # tiempo de simulación
N = 40 # número de puntos en la discretización espacial
M = 400 # número de puntos en la discretización temporal
alpha = 0.3 # coeficiente de difusión

# Discretización
dx = b / N
dt = d / M
x = np.linspace(0, b, N+1)
t = np.linspace(0, d, M+1)

# Condición inicial
u = np.zeros((N+1, M+1))
u[:N//2, 0] = 1  # f(x) = 1 para 0 < x < b/2

# Coeficiente de la ecuación discretizada
coef = alpha * dt / dx**2

# Implementación del método de Gauss-Seidel para la ecuación de calor
for j in range(0, M):
    for it in range(100):  # Número de iteraciones para la convergencia
        u_old = u[:, j+1].copy()
        for i in range(1, N):
            u[i, j+1] = u[i, j] + coef * (u[i-1, j+1] - 2*u[i, j] + u[i+1, j])
        # Condiciones de contorno: u(0, t) = u(b, t) = 0
        u[0, j+1] = u[N, j+1] = 0
        # Criterio de convergencia
        if np.linalg.norm(u[:, j+1] - u_old, np.inf) < 1e-5:
            break

# Preparando los datos para la representación 3D
X, T = np.meshgrid(x, t)
U = u.T  # Transponer para alinear con las dimensiones de la malla

# Representación 3D
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, T, U, cmap='viridis')

ax.set_xlabel('Posición x')
ax.set_ylabel('Tiempo t')
ax.set_zlabel('Temperatura u(x,t)')
ax.set_title('Evolución de la Temperatura en la Barra')
plt.tight_layout()
plt.show()
