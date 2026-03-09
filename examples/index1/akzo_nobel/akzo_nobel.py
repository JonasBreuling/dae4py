import numpy as np
from dae4py.dae_problem import DAEProblem

k1 = 18.7
k2 = 0.58
k3 = 0.09
k4 = 0.42
K = 34.4
Ks = 115.83
klA = 3.3
H = 737
pCO2 = 0.9


def f(y, z):
    y1, y2, y3, y4, y5 = y

    # r1 = k1 * y1**2 * np.sqrt(y2) # original
    r1 = k1 * y1**4 * np.sqrt(y2) # modified
    r2 = k2 * y3 * y4
    r3 = k2 / K * y5 * y1
    # r4 = k3 * y1 * y4**2 * y2 # original
    r4 = k3 * y1 * y4**2 # modified
    # r5 = k4 * z * np.sqrt(y2) # original
    r5 = k4 * z**2 * np.sqrt(y2) # modified
    Fin = klA * (pCO2 / H - y2)

    yp = np.zeros_like(y)
    yp[0] = -2 * r1 + r2 - r3 - r4
    yp[1] = -0.5 * r1 - r4 - 0.5 * r5 + Fin
    yp[2] = r1 - r2 + r3
    yp[3] = -r2 + r3 - 2 * r4
    yp[4] = r2 - r3 + r5

    return yp


def g(y, z):
    y1, y2, y3, y4, y5 = y
    return Ks * y1 * y4 - z


def F(t, x, xp):
    y = x[:-1]
    yp = xp[:-1]
    z = x[-1]

    F = np.zeros(6, dtype=np.common_type(y, yp))
    F[:-1] = yp - f(y, z)
    F[-1] = g(y, z)

    return F


t0 = 0
t1 = 180
y0 = np.array([0.444, 0.00123, 0, 0.007, 0], dtype=float)
z0 = Ks * y0[0] * y0[3]
yp0 = f(y0, z0)
zp0 = 0.0

x0 = np.array([*y0, z0])
xp0 = np.array([*yp0, zp0])

F0 = F(t0, x0, xp0)
print(f"F0: {F0}")

problem = DAEProblem(
    name="Akzo Nobel",
    F=F,
    t_span=(t0, t1),
    index=1,
    y0=x0,
    yp0=xp0,
)
