import numpy as np
from dae4py.dae_problem import DAEProblem

INDEX = 2  # possible options: [1, 2]

m = 1
Theta = 1
g = 9.81
Omega = 1
alpha = 35 / 180 * np.pi
salpha = np.sin(alpha)


match INDEX:
    case 1:

        def F(t, vy, vyp):
            x, y, phi, u, v, omega, _ = vy
            xp, yp, phip, up, vp, omegap, lap = vyp

            sphi, cphi = np.sin(phi), np.cos(phi)

            F = np.zeros_like(vy, dtype=np.common_type(vy, vyp))
            F[0] = xp - u
            F[1] = yp - v
            F[2] = phip - omega
            F[3] = m * up + cphi * lap
            F[4] = m * vp - sphi * lap - m * g * salpha
            F[5] = Theta * omegap
            F[6] = v * sphi - u * cphi
            return F

        def true_sol(t):
            x = (g * salpha / Omega) * (t / 2 - np.sin(2 * Omega * t) / (4 * Omega))
            y = (g * salpha / (2 * Omega**2)) * np.sin(Omega * t) ** 2
            phi = Omega * t

            u = (g * salpha / Omega) * np.sin(Omega * t) ** 2
            v = (g * salpha / Omega) * np.sin(Omega * t) * np.cos(Omega * t)
            omega = Omega * np.ones_like(t)

            La = (2 * m * g * salpha / Omega) * np.cos(Omega * t)
            La_dot = -2 * m * g * salpha * np.sin(Omega * t)

            x_dot = u
            y_dot = v
            phi_dot = omega

            u_dot = -np.cos(phi) * La_dot / m
            v_dot = g * salpha + np.sin(phi) * La_dot / m
            omega_dot = np.zeros_like(t)

            vy = np.array(
                [
                    x,
                    y,
                    phi,
                    u,
                    v,
                    omega,
                    La,
                ]
            )

            vyp = np.array(
                [
                    x_dot,
                    y_dot,
                    phi_dot,
                    u_dot,
                    v_dot,
                    omega_dot,
                    La_dot,
                ]
            )

            return vy, vyp

    case 2:

        def F(t, vy, vyp):
            x, y, phi, u, v, omega, la = vy
            xp, yp, phip, up, vp, omegap, _ = vyp

            sphi, cphi = np.sin(phi), np.cos(phi)

            F = np.zeros_like(vy, dtype=np.common_type(vy, vyp))
            F[0] = xp - u
            F[1] = yp - v
            F[2] = phip - omega
            F[3] = m * up + cphi * la
            F[4] = m * vp - sphi * la - m * g * salpha
            F[5] = Theta * omegap
            F[6] = v * sphi - u * cphi
            return F

        def true_sol(t):
            x = (g * salpha / Omega) * (t / 2 - np.sin(2 * Omega * t) / (4 * Omega))
            y = (g * salpha / (2 * Omega**2)) * np.sin(Omega * t) ** 2
            phi = Omega * t

            u = (g * salpha / Omega) * np.sin(Omega * t) ** 2
            v = (g * salpha / Omega) * np.sin(Omega * t) * np.cos(Omega * t)
            omega = Omega * np.ones_like(t)

            la = -2 * m * g * salpha * np.sin(Omega * t)
            la_dot = -2 * m * g * salpha * Omega * np.cos(Omega * t)

            x_dot = u
            y_dot = v
            phi_dot = omega

            u_dot = -np.cos(phi) * la / m
            v_dot = g * salpha + np.sin(phi) * la / m
            omega_dot = np.zeros_like(t)

            vy = np.array(
                [
                    x,
                    y,
                    phi,
                    u,
                    v,
                    omega,
                    la,
                ]
            )

            vyp = np.array(
                [
                    x_dot,
                    y_dot,
                    phi_dot,
                    u_dot,
                    v_dot,
                    omega_dot,
                    la_dot,
                ]
            )

            return vy, vyp


problem = DAEProblem(
    name="Knife edge",
    F=F,
    t_span=(0, 2 * np.pi / Omega),
    index=INDEX,
    true_sol=true_sol,
)
