import time
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
from dae4py.radau import solve_dae_radau
from scipy.integrate import solve_ivp


def ax2skew(a):
    """Computes the skew symmetric matrix from a 3D vector."""
    assert a.size == 3
    # fmt: off
    return np.array([[0,    -a[2], a[1] ],
                     [a[2],  0,    -a[0]],
                     [-a[1], a[0], 0    ]], dtype=a.dtype)
    # fmt: on


def Exp_SO3_quat(P, normalize=True):
    """Exponential mapping defined by (unit) quaternion, see 
    Egeland2002 (6.199) and Nuetzi2016 (3.31).

    References:
    -----------
    Egenland2002: https://folk.ntnu.no/oe/Modeling%20and%20Simulation.pdf \\
    Nuetzi2016: https://www.research-collection.ethz.ch/handle/20.500.11850/117165
    """
    p0, p = np.array_split(P, [1])
    p_tilde = ax2skew(p)
    if normalize:
        P2 = P @ P
        return np.eye(3, dtype=P.dtype) + (2 / P2) * (p0 * p_tilde + p_tilde @ p_tilde)
    else:
        return np.eye(3, dtype=P.dtype) + 2 * (p0 * p_tilde + p_tilde @ p_tilde)


def Spurrier(A):
    """
    Spurrier's algorithm to extract the unit quaternion from a given rotation
    matrix, see Spurrier19978, Simo1986 Table 12 and Crisfield1997 Section 16.10.

    References
    ----------
    Spurrier19978: https://arc.aiaa.org/doi/10.2514/3.57311 \\
    Simo1986: https://doi.org/10.1016/0045-7825(86)90079-4 \\
    Crisfield1997: http://inis.jinr.ru/sl/M_Mathematics/MN_Numerical%20methods/MNf_Finite%20elements/Crisfield%20M.A.%20Vol.2.%20Non-linear%20Finite%20Element%20Analysis%20of%20Solids%20and%20Structures..%20Advanced%20Topics%20(Wiley,1996)(ISBN%20047195649X)(509s).pdf
    """
    decision = np.zeros(4, dtype=float)
    decision[:3] = np.diag(A)
    decision[3] = np.trace(A)
    i = np.argmax(decision)

    quat = np.zeros(4, dtype=float)
    if i != 3:
        j = (i + 1) % 3
        k = (j + 1) % 3

        quat[i + 1] = np.sqrt(0.5 * A[i, i] + 0.25 * (1 - decision[3]))
        quat[0] = (A[k, j] - A[j, k]) / (4 * quat[i + 1])
        quat[j + 1] = (A[j, i] + A[i, j]) / (4 * quat[i + 1])
        quat[k + 1] = (A[k, i] + A[i, k]) / (4 * quat[i + 1])

    else:
        quat[0] = 0.5 * np.sqrt(1 + decision[3])
        quat[1] = (A[2, 1] - A[1, 2]) / (4 * quat[0])
        quat[2] = (A[0, 2] - A[2, 0]) / (4 * quat[0])
        quat[3] = (A[1, 0] - A[0, 1]) / (4 * quat[0])

    return quat


class RigidBody:
    def __init__(self, mass, B_Theta_C, r_OC0, A_IB0, v_C0, B_Omega_IB0, stabilize=None):
        self.mass = mass
        self.B_Theta_C = B_Theta_C
        
        self.q0 = RigidBody.pose2q(r_OC0, A_IB0)
        self.u0 = np.concatenate([v_C0, B_Omega_IB0])

        self.stabilize = stabilize
        if stabilize in [None, "project", "elastic"]:
            self.y0 = np.array([*self.q0, *self.u0])
            self.yp0 = np.array([*self.q_dot(self.q0, self.u0), *self.u_dot(self.q0, self.u0)])
        elif stabilize == "multiplier":
            self.y0 = np.array([*self.q0, *self.u0, 0])
            self.yp0 = np.array([*self.q_dot(self.q0, self.u0), *self.u_dot(self.q0, self.u0), 0])
        else:
            raise NotImplementedError(f"stabilization: {stabilize} is not implemented")
        
        if stabilize == "project":
            self.events = [self.__event]
        else:
            self.events = None

    @staticmethod
    def pose2q(r_OC, A_IB):
        return np.concatenate([r_OC, Spurrier(A_IB)])
    
    def q_dot(self, q, u):
        r_OC, P = np.array_split(q, [3])
        p0, p = np.array_split(P, [1])
        v_C, B_Omega_IB = np.array_split(u, [3])

        q_dot = np.zeros(7)
        q_dot[:3] = v_C
        q_dot[3:] = 0.5 * np.vstack((-p.T, p0 * np.eye(3, dtype=P.dtype) + ax2skew(p))) @ B_Omega_IB
        return q_dot
    
    def u_dot(self, q, u):
        r_OC, P = np.array_split(q, [3])
        p0, p = np.array_split(P, [1])
        v_C, B_Omega_IB = np.array_split(u, [3])

        u_dot = np.zeros(6)
        u_dot[:3] = 0 
        u_dot[3:] = np.linalg.solve(self.B_Theta_C, -ax2skew(B_Omega_IB) @ self.B_Theta_C @ B_Omega_IB)

        return u_dot
    
    def __call__(self, t, y, yp):
        if self.stabilize in [None, "project", "elastic"]:
            q, u = np.array_split(y, [7])
            q_dot, u_dot = np.array_split(yp, [7])
        elif self.stabilize == "multiplier":
            q, u, mu = np.array_split(y, [7, 13])
            q_dot, u_dot, mu_dot = np.array_split(yp, [7, 13])

        r_OC, P = np.array_split(q, [3])
        p0, p = np.array_split(P, [1])
        v_C, B_omega_IB = np.array_split(u, [3])

        r_OC_dot, P_dot = np.array_split(q_dot, [3])
        a_C, B_psi_IB = np.array_split(u_dot, [3])

        # residual
        F = np.zeros_like(y)

        # kinematic differential equation
        F[:7] = q_dot - self.q_dot(q, u)

        # stabilize quaternion
        if self.stabilize == "elastic":
            c = 5e-1
            # c = 10
            F[3:7] -= 2 * P * c * (1 - P @ P)
        if self.stabilize == "multiplier":
            F[3:7] -= 2 * P * mu_dot
            F[13] = P.dot(P) - 1.0

        # equations of motion
        if self.stabilize == "multiplier":
            F[7:13] = u_dot - self.u_dot(q, u)
        else:
            F[7:] = u_dot - self.u_dot(q, u)

        return F
    
    def rhs(self, t, y):
        if self.stabilize in [None, "project", "elastic"]:
            q, u = np.array_split(y, [7])
        r_OC, P = np.array_split(q, [3])
        p0, p = np.array_split(P, [1])
        v_C, B_Omega_IB = np.array_split(u, [3])

        rhs = np.zeros_like(y)
        # rhs[:3] = u[:3]
        # rhs[3:7] = 0.5 * np.vstack((-p.T, p0 * np.eye(3, dtype=P.dtype) + ax2skew(p))) @ B_Omega_IB
        rhs[:7] = self.q_dot(q, u)
        if self.stabilize == "elastic":
            c = 5e-1
            rhs[3:7] += 2 * P * c * (1 - P @ P) #/ (P @ P)
        
        # rhs[7:10] = 0
        # rhs[10:] = np.linalg.solve(self.B_Theta_C, -ax2skew(B_Omega_IB) @ self.B_Theta_C @ B_Omega_IB)
        rhs[7:] = self.u_dot(q, u)
        return rhs


    def __event(self, t, y, yp):
    # def __event(self, t, y):
        # project quaternion to be of unit length
        y[3:7] = y[3:7].copy() / np.linalg.norm(y[3:7].copy())

        return 1

if __name__ == "__main__":
    ############
    # parameters
    ############

    # simulation parameters
    t0 = 0  # initial time
    t1 = 20  # final time

    # inertia properties
    mass = 1
    A = 3
    B = 2
    C = 1
    B_Theta_C = np.diag([A, B, C])

    # initial conditions
    r_OC0 = np.zeros(3)
    A_IB0 = np.eye(3)
    v_C0 = np.zeros(3)
    epsilon = 1e-6
    omega_dot0 = 20
    B_Omega_IB0 = np.array((epsilon, omega_dot0, 0))

    # stabilize = [None, "elastic", "multiplier", "project"]
    # stabilize = [None]
    # stabilize = [None, "elastic"]
    # stabilize = ["elastic"]
    stabilize = [None, "elastic", "multiplier"]
    # stabilize = [None]
    # stabilize = [None, "project"]
    sols = []
    rigid_bodies = []

    for stab in stabilize:
        rigid_body = RigidBody(mass, B_Theta_C, r_OC0, A_IB0, v_C0, B_Omega_IB0, stabilize=stab)
        rigid_bodies.append(rigid_body)

        # initial conditions
        y0 = rigid_body.y0
        # # yp0 = np.zeros_like(rigid_body.y0)
        yp0 = rigid_body.yp0
        # F0 = rigid_body(t0, y0, yp0)
        # # print(f"y0: {y0}")
        # # print(f"yp0: {yp0}")
        # # print(f"F0: {F0}")
        # # exit()

        # y0, yp0, F0 = consistent_initial_conditions(rigid_body, t0, rigid_body.y0, np.zeros_like(rigid_body.y0), fixed_y0=np.arange(6))
        # y0, yp0, F0 = consistent_initial_conditions(rigid_body, t0, rigid_body.y0, np.zeros_like(rigid_body.y0))
        # print(f"y0: {y0}")
        # print(f"yp0: {yp0}")
        # print(f"F0: {F0}")

        ##############
        # solver setup
        ##############
        t_span = (t0, t1)
        # t_eval = np.linspace(t0, t1, num=int(1e3))
        t_eval = None

        atol = 1e-6
        rtol = 1e-4
        max_step = 1e-1
        max_step = np.inf

        ##############
        # dae solution
        ##############
        start = time.time()
        sol = solve_dae_radau(rigid_body, y0, yp0, t_span, t_eval=t_eval, rtol=rtol, atol=atol, max_step=max_step)
        end = time.time()
        print(f"- stabilize: {stab}; elapsed time: {end - start}")
        print(sol)
        sols.append(sol)

    #################
    # post-processing
    #################
    for stab, sol, rigid_body in zip(stabilize, sols, rigid_bodies):
        t = sol.t
        y = sol.y.copy()
        q, u = np.array_split(y, [7], axis=1)
        
        r_OC = np.array([qi[:3] for qi in q])
        P = np.array([qi[3:] for qi in q])
        e_zB = np.array([Exp_SO3_quat(qi[3:])[:, 1] for qi in q])

        omega = np.array([ui[3:6] for ui in u])

        ########
        # export
        ########
        header = "t, x, y, z, e_y1, e_y2, e_y3, p0, p1, p2, p3, omega_x, omega_y, omega_z, norm_P"

        data = np.concatenate((
            t[:, None],
            r_OC,
            e_zB,
            P,
            omega,
            np.linalg.norm(P, axis=1)[:, None],
        ), axis=1)

        path = Path(sys.modules["__main__"].__file__)

        np.savetxt(
            path.parent / (path.stem + f"_{stab}.txt"),
            data,
            delimiter=", ",
            header=header,
            comments="",
        )

    ###############
    # visualization
    ###############
    fig, ax = plt.subplots(4, 1, figsize=(10, 7))
    for stab, sol, rigid_body in zip(stabilize, sols, rigid_bodies):
        t = sol.t
        y = sol.y.copy()
        if rigid_body.stabilize in [None, "project", "elastic"]:
            q, u = np.array_split(y, [7], axis=1)
        elif rigid_body.stabilize == "multiplier":
            q, u, mu = np.array_split(y, [7, 13], axis=1)

        r_OC, P = np.array_split(q, [3], axis=1)

        ax[0].set_title("omega_x")
        ax[0].plot(t, u[:, 3], label=f"{stab}")

        ax[1].set_title("omega_y")
        ax[1].plot(t, u[:, 4], label=f"{stab}")

        ax[2].set_title("omega_z")
        ax[2].plot(t, u[:, 5], label=f"{stab}")

        ax[3].set_title("Quaternion length")
        ax[3].plot(t, np.linalg.norm(P, axis=1), label=f"$\|P\|$ ({stab})")

    ax[0].set_xlabel("t")
    ax[0].grid()
    ax[0].legend()

    ax[1].set_xlabel("t")
    ax[1].grid()
    ax[1].legend()

    ax[2].set_xlabel("t")
    ax[2].grid()
    ax[2].legend()

    ax[3].set_xlabel("t")
    ax[3].grid()
    ax[3].legend()

    plt.tight_layout()
    plt.show()
