import numpy as np
import sympy as sp
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def Rot2D(X, Y, Alpha):
    RX = X*np.cos(Alpha) - Y*np.sin(Alpha)
    RY = X*np.sin(Alpha) + Y*np.cos(Alpha)
    return RX, RY

R = 4
Omega = 1

T = np.linspace(0, 10, 50)

t = sp.Symbol('t')

x = R*(Omega*t - sp.sin(Omega*t))
y = R*(1-sp.cos(Omega*t))

xC = R*Omega*t

Vx = sp.diff(x, t)
Vy = sp.diff(y, t)
v = (Vx ** 2 + Vy ** 2) ** 0.5

Wx = sp.diff(Vx, t)
Wy = sp.diff(Vy, t)
w = (Wx ** 2 + Wy ** 2) ** 0.5
Wtan = sp.diff(v, t)
Wnor = (w ** 2 - Wtan ** 2) ** 0.5

ro = v ** 2 / Wnor

# Радиус кривизны сонаправлен с нормальным ускорением
# Вектор полного ускорения - это сумма векторов нормального и тангенциального ускорений
# Следовательно, Вектор нормального ускорения - это разность векторов полного и тангенциального ускорений
    # Но у нас нет вектора тангенциального ускорения - только его модуль
    # Но оно сонаправлено со скоростью - отнормируем вектор скорости, умножим на модуль тангенциального ускорения
# Нормируем вектор нормального ускорения, получим вектор n
# Зная направляющий вектор и радиус кривизны сможем отрисовать кривизну

WTanx = Vx / v * Wtan
WTany = Vy / v * Wtan

NotNormNx = Wx - WTanx
NotNormNy = Wy - WTany
NotNormn = (NotNormNx ** 2 + NotNormNy ** 2) ** 0.5

Nx = NotNormNx / NotNormn
Ny = NotNormNy / NotNormn

Curvax = Nx * ro
Curvay = Ny * ro

X = np.zeros_like(T)
Y = np.zeros_like(T)
VX = np.zeros_like(T)
VY = np.zeros_like(T)
WX = np.zeros_like(T)
WY = np.zeros_like(T)
XC = np.zeros_like(T)
YC = R
RO = np.zeros_like(T)
CurvaX = np.zeros_like(T)
CurvaY = np.zeros_like(T)

for i in np.arange(len(T)):
    X[i] = sp.Subs(x, t, T[i])
    Y[i] = sp.Subs(y, t, T[i])
    VX[i] = sp.Subs(Vx, t, T[i])
    VY[i] = sp.Subs(Vy, t, T[i])
    WX[i] = sp.Subs(Wx, t, T[i])
    WY[i] = sp.Subs(Wy, t, T[i])
    XC[i] = sp.Subs(xC, t, T[i])
    RO[i] = sp.Subs(ro, t, T[i])
    CurvaX[i] = sp.Subs(Curvax, t, T[i])
    CurvaY[i] = sp.Subs(Curvay, t, T[i])

fig = plt.figure()

ax1 = fig.add_subplot(1, 1, 1)
ax1.axis('equal')
ax1.set(xlim=[-R, 12*R], ylim=[-R, 3*R])

ax1.plot(X, Y)

ax1.plot([X.min(), X.max()], [0, 0], 'black')

P, = ax1.plot(X[0], Y[0], marker='o')
RLine, = ax1.plot([0, X[0]], [0, Y[0]], 'm')
VLine, = ax1.plot([X[0], X[0]+VX[0]], [Y[0], Y[0]+VY[0]], 'r')
WLine, = ax1.plot([X[0], X[0]+WX[0]], [Y[0], Y[0]+WY[0]], 'g')
CurvaLine, = ax1.plot([X[0], X[0]+CurvaX[0]], [Y[0], Y[0]+CurvaY[0]], 'c')

ArrowX = np.array([-0.2*R, 0, -0.2*R])
ArrowY = np.array([0.1*R, 0, -0.1*R])

RArrowX, RArrowY = Rot2D(ArrowX, ArrowY, math.atan2(Y[0], X[0]))
RArrow, = ax1.plot(RArrowX+X[0], RArrowY+Y[0], 'r')

RArrowX, RArrowY = Rot2D(ArrowX, ArrowY, math.atan2(VY[0], VX[0]))
VArrow, = ax1.plot(RArrowX+X[0]+VX[0], RArrowY+Y[0]+VY[0], 'r')

RArrowX, RArrowY = Rot2D(ArrowX, ArrowY, math.atan2(WY[0], WX[0]))
WArrow, = ax1.plot(RArrowX+X[0]+WX[0], RArrowY+Y[0]+WY[0], 'g')

RArrowX, RArrowY = Rot2D(ArrowX, ArrowY, math.atan2(CurvaY[0], CurvaX[0]))
CurvaArrow, = ax1.plot(RArrowX+X[0]+CurvaX[0], RArrowY+Y[0]+CurvaY[0], 'c')

Phi = np.linspace(0, 2*math.pi, 100)
Circ, = ax1.plot(XC[0]+R*np.cos(Phi), YC+R*np.sin(Phi), 'b')
CircCurva, = ax1.plot(X[0] + CurvaX[0] + RO[0] * np.cos(Phi), Y[0] + CurvaY[0] + RO[0] * np.sin(Phi), 'c')

def anima(i):
    # print(f"{i}: Vx = {Vx}, Vy = {Vy}, Wx = {Wx}, Wy = {Wy}")
    P.set_data([X[i]], [Y[i]])
    RLine.set_data([0, X[i]], [0, Y[i]])
    VLine.set_data([X[i], X[i]+VX[i]], [Y[i], Y[i]+VY[i]])
    WLine.set_data([X[i], X[i]+WX[i]], [Y[i], Y[i]+WY[i]])
    CurvaLine.set_data([X[i], X[i]+CurvaX[i]], [Y[i], Y[i]+CurvaY[i]])
    
    RArrowX, RArrowY = Rot2D(ArrowX, ArrowY, math.atan2(Y[i], X[i]))
    RArrow.set_data(RArrowX+X[i], RArrowY+Y[i])

    RArrowX, RArrowY = Rot2D(ArrowX, ArrowY, math.atan2(VY[i], VX[i]))
    VArrow.set_data(RArrowX+X[i]+VX[i], RArrowY+Y[i]+VY[i])
    
    RArrowX, RArrowY = Rot2D(ArrowX, ArrowY, math.atan2(WY[i], WX[i]))
    WArrow.set_data(RArrowX+X[i]+WX[i], RArrowY+Y[i]+WY[i])

    RArrowX, RArrowY = Rot2D(ArrowX, ArrowY, math.atan2(CurvaY[i], CurvaX[i]))
    CurvaArrow.set_data(RArrowX+X[i]+CurvaX[i], RArrowY+Y[i]+CurvaY[i])
    
    Circ.set_data(XC[i]+R*np.cos(Phi), YC+R*np.sin(Phi))
    CircCurva.set_data(X[i] + CurvaX[i] + RO[i] * np.cos(Phi), Y[i] + CurvaY[i] + RO[i] * np.sin(Phi))
    return P, VLine, VArrow, WArrow, CurvaArrow, Circ, CircCurva

anim = FuncAnimation(fig, anima, frames=50, interval=200, repeat=False)

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

plt.show()