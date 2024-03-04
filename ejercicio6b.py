import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parámetros de la simulación
N = 20  # Tamaño de la malla (N+2 x N+2 para incluir las fronteras)
lambda_sq = 300**2  # Valor de lambda al cuadrado
tol = 1e-4  # Tolerancia para la convergencia
max_iter = 5000  # Número máximo de iteraciones
dx = 1.0 / (N + 1)  # Distancia entre puntos en la malla

# Inicialización de la malla con ceros (incluyendo las condiciones de contorno)
u = np.zeros((N+2, N+2))

# Aplicar las condiciones de contorno
u[-1, :] = 1  # u(1, y) = 1

# Implementación del método de Gauss-Seidel
def gauss_seidel_helmholtz(u, lambda_sq, tol, max_iter, dx):
    for it in range(max_iter):
        u_old = u.copy()
        
        for i in range(1, N+1):
            for j in range(1, N+1):
                # Actualización de Gauss-Seidel para la ecuación de Helmholtz
                u[i, j] = ((u_old[i+1, j] + u[i-1, j] + u_old[i, j+1] + u[i, j-1]) - dx**2 * lambda_sq * u_old[i, j]) / (4.0 + dx**2 * lambda_sq)
                
        # Criterio de convergencia
        diff = np.max(np.abs(u - u_old))
        if diff < tol:
            print(f"Convergencia alcanzada después de {it+1} iteraciones.")
            return u
    print("Número máximo de iteraciones alcanzado.")
    return u

# Resolver la ecuación
u_sol = gauss_seidel_helmholtz(u, lambda_sq, tol, max_iter, dx)

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
ax.set_title('Solución 3D de la Ecuación de Helmholtz con λ=300 y Método de Gauss-Seidel')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('U')
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
