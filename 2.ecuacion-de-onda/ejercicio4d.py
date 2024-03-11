# d: queremos formar ondas no estacionarias (f=g=0). ¿como lo harias?

import matplotlib.pyplot as plt
import numpy as np

# Parámetros del dominio y la malla
b = float(input("Ingrese la longitud de la cuerda (b): "))
d = float(input("Ingrese el tiempo (d): "))
N, M = int(input("Número de puntos en la dirección x (N): ")), int(input("Número de puntos en la dirección y (M): "))
h = b / N
k = d / M
v = float(input("Valor de velocidad: "))  
p = v * k / h

# Inicializar la matriz de solución
w = np.zeros((M + 1, N + 1))

# Funciones de condiciones iniciales y de contorno
def f(x):
    return 0

def g(x):
    return 0

# Condiciones de contorno
for i in range(1, N):
    w[0][i] = f(h * i)
    w[1][i] = w[0][i] + k * g(h * i)

for j in range(1, M):
    w[j][0] = np.sin(k*j)
    w[j][N] = 0

# Método de diferencias finitas
for j in range(1, M):
    for i in range(1, N):
        w[j+1][i] = 2 * (1 - p*2) * w[j][i] - w[j-1][i] + p*2 * (w[j][i+1] + w[j][i-1])

# Crear coordenadas para el gráfico 3D
X = np.linspace(0, b, N+1)
Y = np.linspace(0, d, M+1)
X, Y = np.meshgrid(X, Y)

# Crear gráfico 3D
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Graficar la solución en 3D
ax.plot_surface(X, Y, w, cmap='viridis')

# Configuraciones adicionales del gráfico
ax.set_xlabel('Posición a lo largo de la cuerda (m)')
ax.set_ylabel('Tiempo (s)')
ax.set_zlabel('Desplazamiento de la cuerda (m)')

# Mostrar el gráfico
plt.show()