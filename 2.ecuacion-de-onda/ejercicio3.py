# g(x)=sen(x), b=pi, d = 10, n = 40, m = 400, v = 3

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Updating parameters for the new conditions
b = np.pi  # Length of the string is now π
d = 10  # Total simulation time remains 10
n = 40  # Spatial steps remains 40
m = 400  # Temporal steps remains 400
v = 3

# Calculating Δx, Δt, and r again
Δx = b / n
Δt = d / m
r = v * Δt / Δx

# Initializing the u matrix for storing string positions with new initial conditions
u = np.zeros((n+1, m+1))
x = np.linspace(0, b, n+1)

# Setting the initial condition according to the new f(x) and g(x)
u[:, 0] = np.sin(x)  # f(x) = sin(x)
# For g(x) = sin(x), we'll approximate the first time step based on the velocity condition
# Assuming a small initial velocity, we approximate this by modifying the second time step
u[:, 1] = u[:, 0] + Δt * np.sin(x)

# Applying the finite difference method with the new initial conditions
for j in range(1, m):
    for i in range(1, n):
        u[i, j+1] = (2 - 2*r**2)*u[i, j] + r**2*(u[i+1, j] + u[i-1, j]) - u[i, j-1]

# Plotting the string in 3D with the new conditions
X, Y = np.meshgrid(x, np.linspace(0, d, m+1))
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plotting the surface
ax.plot_surface(X, Y, u.T, cmap='viridis')

ax.set_title('Evolución de la cuerda en el tiempo con f(x)=sin(x) en 3D')
ax.set_xlabel('Posición a lo largo de la cuerda (m)')
ax.set_ylabel('Tiempo (s)')
ax.set_zlabel('Desplazamiento de la cuerda (m)')
plt.show()
