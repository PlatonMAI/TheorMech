import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math
from scipy.integrate import odeint

def odesys(y, t, R, l, m1, m2, g, M0, gamma, k):
    dy = np.zeros(4)
    dy[0] = y[2]
    dy[1] = y[3]

    sqrt_ = math.sqrt(R**2 - l**2)
    sin_ = np.sin(y[1] - y[0])
    cos_ = np.cos(y[1] - y[0])

    a11 = (2*m1 + m2)*R**2
    a12 = m2*R*sqrt_*cos_
    a21 = sqrt_*R*cos_
    a22 = R**2 - 2/3*l**2

    b1 = m2*R*sqrt_*y[3]**2*sin_ - (m1 + m2)*g*R*np.sin(y[0]) + M0*np.sin(gamma*t) - k*y[2]
    b2 = -sqrt_*(R*y[2]**2*sin_ + g*np.sin(y[1]))

    dy[2] = (b1*a22 - b2*a12)/(a11*a22 - a12*a21)
    dy[3] = (b2*a11 - b1*a21)/(a11*a22 - a12*a21)

    return dy

# Константы
R = 0.5
l = 0.25
m1 = 2
m2 = 1
g = 9.81
M0 = 15
gamma = 3/2*math.pi
k = 10

# Вспомогательные
forLetters = 0.01
TriangleLength = R * 0.5

# Кадры
steps = 1001
t_fin = 20
t = np.linspace(0, t_fin, steps)

# Начальные условия
phi0 = 0
psi0 = math.pi / 6
dphi0 = 0
dpsi0 = 0
y0 = [phi0, psi0, dphi0, dpsi0]

# Проинтегрированная система
Y = odeint(odesys, y0, t, (R, l, m1, m2, g, M0, gamma, k))

# Функции
phi = Y[:,0]
psi = Y[:,1]

dphi = Y[:,2]
dpsi = Y[:,3]
ddphi = [odesys(y, t, R, l, m1, m2, g, M0, gamma, k)[2] for y,t in zip(Y,t)]
ddpsi = [odesys(y, t, R, l, m1, m2, g, M0, gamma, k)[3] for y,t in zip(Y,t)]

NO1X = -(m1 + m2)*R*(ddphi*np.cos(phi) - dphi**2*np.sin(phi)) - m2*math.sqrt(R**2 - l**2)*(ddpsi*np.cos(psi) - dpsi**2*np.sin(psi))
NO1Y = -(m1 + m2)*R*((ddphi*np.sin(phi) + dphi**2*np.cos(phi)) + g) - m2*math.sqrt(R**2 - l**2)*(ddpsi*np.sin(psi) + dpsi**2*np.cos(psi))
NO1 = np.sqrt(NO1X**2 + NO1Y**2)

# Вычисленные константы
alpha = -math.acos(l / R)
OC = -R * math.sin(alpha)

# Задаем положение
O1X = 2 * R + R / 10
O1Y = 2 * R + R / 10

# относительно т. О1
OX = O1X + R * np.sin(phi)
OY = O1Y - R * np.cos(phi)
# относительно т. О
CX = OX + OC * np.sin(psi)
CY = OY - OC * np.cos(psi)
AX = OX + R * np.cos(alpha + psi)
AY = OY + R * np.sin(alpha + psi)
BX = OX + R * np.cos(math.pi - alpha + psi)
BY = OY + R * np.sin(math.pi - alpha + psi)



fig = plt.figure()

ax1 = fig.add_subplot()
ax1.axis('equal')
plt.gca().set_adjustable("box")
ax1.set(xlim=[0, 4 * R + 2 * R / 10], ylim=[0, 2 * R + TriangleLength + 2 * R / 10])

ax1.plot(O1X, O1Y, marker = 'o')
plt.text(O1X + forLetters, O1Y  + forLetters, 'O1')

OPoint = ax1.plot(OX[0], OY[0], marker = 'o')[0]
APoint = ax1.plot(AX[0], AY[0], marker = 'o')[0]
BPoint = ax1.plot(BX[0], BY[0], marker = 'o')[0]
CPoint = ax1.plot(CX[0], CY[0], marker = 'o')[0]
OText = plt.text(OX[0] + forLetters, OY[0]  + forLetters, 'O')
AText = plt.text(AX[0] + forLetters, AY[0]  + forLetters, 'A')
BText = plt.text(BX[0] + forLetters, BY[0]  + forLetters, 'B')
CText = plt.text(CX[0] + forLetters, CY[0]  + forLetters, 'C')

ABLine = ax1.plot([ AX[0], BX[0] ], [ AY[0], BY[0] ])[0]

phiForCirc = np.linspace(0, 2*math.pi, 100)
Circ = ax1.plot(OX[0] + R * np.cos(phiForCirc), OY[0] + R * np.sin(phiForCirc))[0]

ax1.plot([O1X - TriangleLength / 2, O1X + TriangleLength / 2], [O1Y + TriangleLength / 2, O1Y + TriangleLength / 2], '000')
ax1.plot([O1X - TriangleLength / 2, O1X], [O1Y + TriangleLength / 2, O1Y], '000')
ax1.plot([O1X, O1X + TriangleLength / 2], [O1Y, O1Y + TriangleLength / 2], '000')



def anima(i):
    OPoint.set_data([OX[i]], [OY[i]])
    APoint.set_data([AX[i]], [AY[i]])
    BPoint.set_data([BX[i]], [BY[i]])
    CPoint.set_data([CX[i]], [CY[i]])
    OText.set_position([OX[i] + forLetters, OY[i]  + forLetters])
    AText.set_position([AX[i] + forLetters, AY[i]  + forLetters])
    BText.set_position([BX[i] + forLetters, BY[i]  + forLetters])
    CText.set_position([CX[i] + forLetters, CY[i]  + forLetters])

    ABLine.set_data([ AX[i], BX[i] ], [ AY[i], BY[i] ])

    Circ.set_data(OX[i] + R * np.cos(phiForCirc), OY[i] + R * np.sin(phiForCirc))

anim = FuncAnimation(fig, anima, frames = steps, interval = 50)

anim_running = True
def onClick(event):
    global anim_running
    if anim_running:
        anim.event_source.stop()
        anim_running = False
    else:
        anim.event_source.start()
        anim_running = True
fig.canvas.mpl_connect('button_press_event', onClick)



fig_for_graphs = plt.figure(figsize=[13,7])
ax_for_graphs = fig_for_graphs.add_subplot(2, 2, 1)
ax_for_graphs.plot(t, phi, color='Blue')
ax_for_graphs.set_title("phi(t)")
ax_for_graphs.set(xlim=[0, t_fin])
ax_for_graphs.grid(True)

ax_for_graphs = fig_for_graphs.add_subplot(2, 2, 3)
ax_for_graphs.plot(t, psi, color='Red')
ax_for_graphs.set_title("psi(t)")
ax_for_graphs.set(xlim=[0, t_fin])
ax_for_graphs.grid(True)

ax_for_graphs = fig_for_graphs.add_subplot(2, 2, 2)
ax_for_graphs.plot(t, NO1, color='Black')
ax_for_graphs.set_title("NO1(t)")
ax_for_graphs.set(xlim=[0, t_fin])
ax_for_graphs.grid(True)

plt.show()