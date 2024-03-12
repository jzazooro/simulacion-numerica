import math
import numpy as np
import matplotlib.pyplot as plt

b=float(input("Ingrese el valor de b: "))
d=float(input("Ingrese el valor de d: "))
N=int(input("Ingrese el valor de n: "))
M=int(input("Ingrese el valor de m: "))
v=float(input("Ingrese el valor de v: "))
h=b/N
k=d/M
r=(v*k)/h

def f(x):
    return math.exp(-(h*i-b/2)**2)

for j in range(1, M):
    W[j][0]=0
    W[j][N]=0

for i in range(1, N):
    W[0][i]=f(h*i)

for j in range(M):
    for i in range(1, N):
        W[j+1][i]=(1-2*k*v**2/h**2)*W[j][i]+k*v**2/     -W[j-1][i]+r**2*(W[j][i+1]+W[j][i-1])


x=np.linspace(0, b, N+1)
y=np.linspace(0, d, M+1)
x, y=np.meshgrid(x, y)

fig=plt.figure()
ax=fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, W, cmap='viridis')

ax.set_xlabel('Posición (x)')
ax.set_ylabel('Tiempo (t)')
ax.set_zlabel('Temperatura (u)')
ax.set_title('Evolución de la temperatura a lo largo del tiempo')
plt.show()

