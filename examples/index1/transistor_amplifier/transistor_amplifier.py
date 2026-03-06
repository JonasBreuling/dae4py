import numpy as np
from dae4py.dae_problem import DAEProblem

# problem parameters
U_b = 6
R0 = 1000
Ri = 9000
R1, R2, R3, R4, R5 = 9000 * np.ones(5)
alpha = 0.99
I_S = 1e-6
U_T = 0.026
U_e = lambda t: 0.4 * np.sin(200 * np.pi * t)
I_E = lambda Delta_U: I_S * (np.exp(Delta_U / U_T) - 1)
C1, C2, C3 = 1e-6 * np.arange(1, 4)

# initial states
y0 = np.zeros(5)
y0[1] = y0[2] = U_b * R1 / (R1 + R2)
y0[3] = U_b

# initial derivatives
yp0 = np.zeros_like(y0)
yp0[2] = (I_E(y0[1] - y0[2]) - y0[2] / R3) / C2


def F(t, y, yp):
    U1, U2, U3, U4, U5 = y
    Up1, Up2, Up3, Up4, Up5 = yp
    I_E_ = I_E(U2 - U3)
    return np.array(
        [
            (U_e(t) - U1) / R0 - C1 * (Up1 - Up2),
            C1 * (Up1 - Up2) - U2 / R1 + (U_b - U2) / R2 - (1 - alpha) * I_E_,
            -U3 / R3 - C2 * Up3 + I_E_,
            C3 * (Up5 - Up4) + (U_b - U4) / R4 - alpha * I_E_,
            -U5 / R5 - C3 * (Up5 - Up4),
        ]
    )


problem = DAEProblem(
    name="Transistor amplifier",
    F=F,
    t_span=(0, 0.2),
    index=1,
    y0=y0,
    yp0=yp0,
    parameters={"Ue": U_e},
)

if __name__ == "__main__":
    # the used set of consistent initial conditions
    F0 = F(0, y0, yp0)
    print(f"y0: {y0}")
    print(f"yp0: {yp0}")
    print(f"F0: {F0}")

    # perturb an algebraic variable and zero initial derivatives
    y0[4] += np.random.rand(1)[0]
    yp0 = np.zeros_like(y0)
    print(f"y0: {y0}")
    print(f"yp0: {yp0}")

    # compute again consistent initial conditions
    from dae4py.consistent_initial_conditions import consistent_initial_conditions

    y0, yp0, F0 = consistent_initial_conditions(F, 0, y0, yp0, fixed_y0=[3])
    print(f"y0: {y0}")
    print(f"yp0: {yp0}")
    print(f"F0: {F0}")
