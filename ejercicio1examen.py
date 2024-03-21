import numpy as np
import math
import matplotlib.pyplot as plt

b= float(input ("b: "))
d= float(input ("d: "))
N= int(input ("N: "))
M= int(input ("M: "))
h= b/N
k= d/M
w= np.zeros((M+1, N+1))
v=float(input("conductividad "))
lamda = (v*2*k)/h*2

def f(x):
    return math.exp(-(x-2.5)**2)

for j in range (1, M): 
    w[j][0]= 0
    w[j][N]= 0
    
for i in range (1, N):
    w[0][i]= f(h*i)
    
for z in range (100):   
    for j in range(M):
        for i in range(1, N):
            w[j][i]= ((k/(h*h))*(w[j][i+1]+w[j][i-1])+(1 + h*i)*(w[j-1][i]))/(1 + h*i + 2*(k/(h*h)))
        
        
        
X= np.linspace(0, b, N+1)
Y= np.linspace(0, d, M+1)
X, Y= np.meshgrid(X, Y)

fig= plt.figure()
ax= fig.add_subplot(111, projection='3d')

ax.plot_surface(X, Y, w, cmap='viridis')

ax.set_xlabel('Eje X')
ax.set_ylabel('Eje Y')

plt.show()