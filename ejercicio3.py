import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ecuacion de poisson

# Parámetros del problema
Lx, Ly = 1, 1  # Dimensiones del dominio
Nx, Ny = 20, 20  # Número de puntos en la malla en cada dirección
dx, dy = Lx / (Nx - 1), Ly / (Ny - 1)  # Distancia entre puntos de la malla

# Inicialización de la malla
u = np.zeros((Nx, Ny))

# Aplicación de las condiciones de contorno
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
u[0, :] = 1 - y**2  # u(0, y) = 1 - y^2
u[-1, :] = 1  # u(1, y) = 1
u[:, 0] = 0  # u(x, 0) = 0
u[:, -1] = x**2  # u(x, 1) = x^2

# Parámetros de la iteración
tolerancia = 1e-4  # Criterio de convergencia
max_iter = 10000  # Número máximo de iteraciones
error = 1  # Inicialización del error
iteracion = 0  # Contador de iteraciones

# Método de Gauss-Seidel
while error > tolerancia and iteracion < max_iter:
    u_prev = u.copy()
    for i in range(1, Nx - 1):
        for j in range(1, Ny - 1):
            u[i, j] = 0.25 * (u[i + 1, j] + u[i - 1, j] + u[i, j + 1] + u[i, j - 1])
    
    # Actualizar las condiciones de contorno en cada iteración
    u[0, :] = 1 - y**2
    u[-1, :] = 1
    u[:, 0] = 0
    u[:, -1] = x**2
    
    error = np.linalg.norm(u - u_prev, ord=np.inf)
    iteracion += 1

# Crear una malla para el ploteo
X, Y = np.meshgrid(x, y)

# Crear la figura y el eje para un gráfico 3D
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Ploteo de la superficie
surf = ax.plot_surface(X, Y, u.T, cmap='viridis', edgecolor='none')

# Etiquetas y título
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('U')
ax.set_title('Solución de ∇u = 0 con condiciones de contorno variadas')

# Barra de colores
fig.colorbar(surf, shrink=0.5, aspect=5)

# Mostrar el gráfico
plt.show()
