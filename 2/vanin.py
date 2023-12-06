import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
# from shapely.geometry import LineString
import math

Steps = 1001
t_fin = 20
t = np.linspace(0, t_fin, Steps)

#phi = np.cos(t) - np.sin(2 * t) NORMIK

#phi = 0.7 * np.cos(2 * t) - 0.7 * np.sin(2 * t) Vase norm

phi = 0.5 * np.cos(2 * t) - 0.5 * np.sin(2 * t)
psi = 0.8 * np.cos(2 * t) - 0.8 * np.sin(2 * t)
#psi = np.linspace(math.pi / 3, math.pi / 3, Steps)
#psi = 0


#CircleX_0 = 8
#CircleY_0 = 4
CircleX_0 = 0
CircleY_0 = 0
R = 2
l = R / 2
R2 = (3) ** 0.5

#Перемещение окружности на угол phi
CircleX = CircleX_0 + R * np.sin(phi)
CircleY = CircleY_0 - R * np.cos(phi)

CircleX2 = (3) ** 0.5 * np.sin(psi)
CircleY2 = (3) ** 0.5 * np.cos(psi)

#Отрисовка окружности
angle = np.linspace(0, np.pi * 2, 150)
X_Circle = R * np.cos(angle)
Y_Circle = R * np.sin(angle)

X_Circle2 = (3) ** 0.5 * np.cos(angle)
Y_Circle2 = (3) ** 0.5 * np.sin(angle)

#X_Triangle = [7, 9, 8, 7]
#Y_Triangle = [7, 7, 6, 7]

#X_Triangle = [-1, 1, 0, -1]
#Y_Triangle = [3, 3, 2, 3]

#Треугольник
X_Triangle = [-1, 1, 0, -1]
Y_Triangle = [1, 1, 0, 1]

#X_A = CircleX_0 - 1
#X_B = CircleX_0 + 1
#Y_A = CircleY_0 - sp.sqrt(R ** 2 - X_A ** 2) - R
#Y_B = CircleY_0 - sp.sqrt(R ** 2 - X_B ** 2) - R

#- R * np.cos(phi)

#X_A = CircleX_0 - 1 + R * np.sin(phi)
#X_B = CircleX_0 + 1 + R * np.sin(phi)  WORK!
#Y_A = CircleY_0 - (3) ** 0.5 - R * np.cos(phi)
#Y_B = CircleY_0 - (3) ** 0.5 - R * np.cos(phi)

O_pointX = CircleX_0 + R * np.sin(phi)
O_pointY = CircleY_0 - R * np.cos(phi)

X_A = CircleX_0 - 1 + R * np.sin(phi) #- l * np.sin(psi)
X_B = CircleX_0 + 1 + R * np.sin(phi) #+ l * np.sin(psi)
Y_A = CircleY_0 - (3) ** 0.5 - R * np.cos(phi) #- l * np.cos(psi)
Y_B = CircleY_0 - (3) ** 0.5 - R * np.cos(phi) #+ l * np.cos(psi)

#C_pointX = X_A + l + 1 * np.sin(psi)
#C_pointY = Y_A #+ 1 * np.cos(psi)

#C_pointX = X_A + l + (3) ** 0.5 * np.sin(psi)
#C_pointY = Y_A + (3) ** 0.5 * np.cos(psi)

C_pointX = O_pointX + R2 * np.sin(psi)
C_pointY = O_pointY - R2 * np.cos(psi)

slope = np.zeros_like(t)
dx = np.zeros_like(t)
dy = np.zeros_like(t)

#Считаем наклон
for i in range(len(t)):
    if ((C_pointX[i] - O_pointX[i]) == 0):
        slope[i] = 0
    else:
        slope[i] = (C_pointY[i] - O_pointY[i]) / (C_pointX[i] - O_pointX[i])

for i in range(len(t)):
    dy[i] = math.sqrt(l**2 / (slope[i] ** 2 + 1))
    dx[i] = -slope[i]*dy[i]

#Строим прямую
X_A2 = C_pointX - dx
Y_A2 = C_pointY - dy
X_B2 = C_pointX + dx
Y_B2 = C_pointY + dy

dlina = np.sqrt((X_B2 - X_A2) ** 2 + (Y_B2 - Y_A2) ** 2)

X_A2N = X_A2 #/ dlina
Y_A2N = Y_A2 #/ dlina
X_B2N = X_B2 #/ dlina
Y_B2N = Y_B2 #/ dlina

fig = plt.figure(figsize = [15, 7])
ax = fig.add_subplot(1, 1, 1)
ax.axis('equal')
ax.set(xlim = [-10, 10], ylim = [-10, 10])

Drawed_Triangle = ax.plot(X_Triangle, Y_Triangle)[0]
Drawed_Circle = ax.plot(CircleX_0 + X_Circle, CircleY_0 + Y_Circle)[0]
#Drawed_Circle2 = ax.plot(CircleX_0 + X_Circle2, CircleY_0 + Y_Circle2)[0]
#Drawed_AB = ax.plot([X_A[0], X_B[0]], [Y_A[0], Y_B[0]], 'g')[0]
Drawed_C = ax.plot(C_pointX[0], C_pointY[0], marker = 'o')[0]
Drawed_O = ax.plot(O_pointX[0], O_pointY[0], marker = 'o')[0]
Drawed_AB2 = ax.plot([X_A2N[0], X_B2N[0]], [Y_A2N[0], Y_B2N[0]], 'b')[0]

def anima(i):
    Drawed_Circle.set_data(X_Circle + CircleX[i], Y_Circle + CircleY[i])
    #Drawed_Circle2.set_data(X_Circle2 + CircleX[i], Y_Circle2 + CircleY[i])
    #Drawed_AB.set_data([X_A[i], X_B[i]], [Y_A[i], Y_B[i]])
    Drawed_C.set_data(C_pointX[i], C_pointY[i])
    Drawed_O.set_data(O_pointX[i], O_pointY[i])
    Drawed_AB2.set_data(([X_A2N[i], X_B2N[i]], [Y_A2N[i], Y_B2N[i]]))
    return Drawed_Circle, Drawed_AB2, Drawed_C, Drawed_O

anim = FuncAnimation(fig, anima, frames = len(t), interval=10, repeat=True)

plt.show()