# visualizar la siguiente solucion de una ecuacion de onda f(x)=x si x<b/2 y f(x)=b-x si x>b/2 b = 5, d = 10, n = 40, m = 400, v = 5

import numpy as np
import matplotlib.pyplot as plt

# ecuacion de onda

b = 5  # Length of the string
d = 10  # Total simulation time
n = 40  # Spatial steps
m = 400  # Temporal steps
v = 5  # Wave velocity

# Calculating Δx, Δt, and r
Δx = b / n
Δt = d / m
r = v * Δt / Δx

# Initializing the u matrix for storing string positions
u = np.zeros((n+1, m+1))
x = np.linspace(0, b, n+1)

# Adjusting the initial condition according to the new f(x)
for i in range(n+1):
    if x[i] < b / 2:
        u[i, 0] = x[i]
    else:
        u[i, 0] = b - x[i]
u[:, 1] = u[:, 0]  # Assuming the first temporal derivative is 0

# Applying the finite difference method with the new initial condition
for j in range(1, m):
    for i in range(1, n):
        u[i, j+1] = (2 - 2*r**2)*u[i, j] + r**2*(u[i+1, j] + u[i-1, j]) - u[i, j-1]

# Plotting the string in 3D with the new initial condition
X, Y = np.meshgrid(x, np.linspace(0, d, m+1))
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plotting the surface
ax.plot_surface(X, Y, u.T, cmap='viridis')

ax.set_title('Evolución de la cuerda en el tiempo con la condición inicial modificada en 3D')
ax.set_xlabel('Posición a lo largo de la cuerda (m)')
ax.set_ylabel('Tiempo (s)')
ax.set_zlabel('Desplazamiento de la cuerda (m)')
plt.show()

