import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ecuacion de poisson

# Parámetros de la simulación
N = 20  # Tamaño de la malla (N+2 x N+2 para incluir las fronteras)
tol = 1e-4  # Tolerancia para la convergencia
max_iter = 10000  # Número máximo de iteraciones

# Inicialización de la malla con ceros (incluyendo las condiciones de contorno)
u = np.zeros((N+2, N+2))

# Implementación del método de Gauss-Seidel
def gauss_seidel(u, tol, max_iter):
    for it in range(max_iter):
        u_old = u.copy()
        
        for i in range(1, N+1):
            for j in range(1, N+1):
                # Actualización de Gauss-Seidel para la ecuación de Laplace
                u[i, j] = 0.25 * (u_old[i+1, j] + u[i-1, j] + u_old[i, j+1] + u[i, j-1])
                
        # Criterio de convergencia
        diff = np.max(np.abs(u - u_old))
        if diff < tol:
            print(f"Convergencia alcanzada después de {it+1} iteraciones.")
            return u
    print("Número máximo de iteraciones alcanzado.")
    return u

# Resolver la ecuación
u_sol = gauss_seidel(u, tol, max_iter)

# Crear la malla de coordenadas
x = np.linspace(0, 1, N+2)
y = np.linspace(0, 1, N+2)
X, Y = np.meshgrid(x, y)

# Gráfico 3D de la solución
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Superficie
surf = ax.plot_surface(X, Y, u_sol, cmap='viridis', edgecolor='none')

# Configuración del gráfico
ax.set_title('Solución 3D de la Ecuación de Laplace con Método de Gauss-Seidel')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('U')
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
