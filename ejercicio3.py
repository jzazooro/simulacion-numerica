import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parámetros del problema
Lx, Ly = 1, 1  # Dimensiones del dominio
Nx, Ny = 20, 20  # Número de puntos en la malla en cada dirección
dx, dy = Lx / (Nx - 1), Ly / (Ny - 1)  # Distancia entre puntos de la malla

# Inicialización de la malla
u = np.zeros((Nx, Ny))

# Condiciones de contorno
u[-1, :] = 1  # u(1, y) = 1

# Definición de x e y para la malla
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)

# Parámetros de la iteración
tolerancia = 1e-4  # Criterio de convergencia
max_iter = 10000  # Número máximo de iteraciones
error = 1  # Inicialización del error
iteracion = 0  # Contador de iteraciones

# Coeficiente adicional debido al término fuente 2u
coef = 2 * dx**2

# Método de Gauss-Seidel para ∇u = 2u
while error > tolerancia and iteracion < max_iter:
    u_prev = u.copy()
    for i in range(1, Nx - 1):
        for j in range(1, Ny - 1):
            u[i, j] = (u[i+1, j] + u[i-1, j] + u[i, j+1] + u[i, j-1]) / (4 + coef)
    
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
ax.set_title('Solución de ∇u = 2u con condiciones de contorno específicas')

# Barra de colores
fig.colorbar(surf, shrink=0.5, aspect=5)

# Mostrar el gráfico
plt.show()
