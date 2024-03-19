import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ecuacion de calor

# Parámetros dados
b = 5       
d = 10      
N = 40      
M = 400     
alpha = 0.3  # Coeficiente de difusión térmica

# Calculando Δx, Δt
dx = b / N  
dt = d / M  

# Malla espacial y temporal
x = np.linspace(0, b, N+1)
t = np.linspace(0, d, M+1)
X, T = np.meshgrid(x, t)

# Condición inicial
u = np.zeros((M+1, N+1))
u[0, :] = np.exp(-((x - b/2)**2))

# Solución numérica
for n in range(0, M):
    for i in range(1, N):
        u[n+1, i] = u[n, i] + alpha*dt/dx**2 * (u[n, i-1] - 2*u[n, i] + u[n, i+1])

# Gráfico 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, T, u, cmap='viridis')

ax.set_xlabel('Posición (x)')
ax.set_ylabel('Tiempo (t)')
ax.set_zlabel('Temperatura (u)')
ax.set_title('Evolución de la Temperatura a lo largo del tiempo')
plt.colorbar(surf)
plt.show()

alphamax=dx/np.sqrt(2*dt)
print("el valo mazimo de alpha es: ", alphamax)