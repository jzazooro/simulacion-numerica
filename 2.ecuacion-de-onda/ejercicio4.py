# f(x)=X[1,3]=0 si x no pertenece a [1, 2] y 1 si x pertenece a [1, 3], b=6, d=24, n=600, m=2400, v=0.5
# X[1, 3] es funcion caracteristica

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Defining parameters for the new setup
b = 6  # Length of the string is now 6
d = 24  # Total simulation time is 24
n = 600  # Spatial steps are 600
m = 2400  # Temporal steps are 2400
v = 0.5  # Wave velocity remains 0.5

# Calculating Δx, Δt, and r
Δx = b / n
Δt = d / m
r = v * Δt / Δx

# Initializing the matrix for storing string positions
u = np.zeros((n+1, m+1))
x = np.linspace(0, b, n+1)

# Setting the initial condition for f(x)
u[:, 0] = np.where((x >= 1) & (x <= 3), 1, 0)
u[:, 1] = u[:, 0]  # Assuming the initial velocity is zero, which simplifies to f(x) for the second step

# Applying the finite difference method
for j in range(1, m):
    for i in range(1, n):
        u[i, j+1] = (2 - 2*r**2)*u[i, j] + r**2*(u[i+1, j] + u[i-1, j]) - u[i, j-1]

# Plotting the string in 3D
X, Y = np.meshgrid(x, np.linspace(0, d, m+1))
fig = plt.figure(figsize=(14, 9))
ax = fig.add_subplot(111, projection='3d')

# Plotting the surface
ax.plot_surface(X, Y, u.T, cmap='viridis')

ax.set_title('Evolución de la cuerda en el tiempo con condición inicial modificada')
ax.set_xlabel('Posición a lo largo de la cuerda')
ax.set_ylabel('Tiempo')
ax.set_zlabel('Desplazamiento de la cuerda')
plt.show()

# b:pregunta de examen, por que la onda se propaga hacia esas 2 rectas verdes?
# respuesta: por que esas 2 rectas verdes son las curvas caracteristicas de la EDP
# c:hallar a mano las curvas caracteristicas de la EDP para ver que da lo mismo
# d: queremos formar ondas no estacionarias (f=g=0). ¿como lo harias?