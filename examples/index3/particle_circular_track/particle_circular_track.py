import numpy as np
from dae4py.dae_problem import DAEProblem

INDEX = "Hiller"  # possible options: [0, 1, 2, 3, "GGL", "Hiller"]

omega = 2 * np.pi


def PHI(t):
    """The time derivative of this function has to be phi_p(t)**2."""
    return omega**2 * (t / 2 + np.sin(2 * t) / 4)


def phi(t):
    return omega * np.sin(t)


def phi_p(t):
    return omega * np.cos(t)


def phi_pp(t):
    """This coincides with the force F(t)."""
    return -omega * np.sin(t)


match INDEX:
    case 0:

        def F(t, vy, vyp):
            x, y, u, v = vy
            x_dot, y_dot, u_dot, v_dot = vyp

            force = phi_pp(t)

            R = np.zeros(4, dtype=np.common_type(vy, vyp))

            la = -0.5 * (u**2 + v**2) / (x**2 + y**2)
            R[0] = x_dot - u
            R[1] = y_dot - v
            R[2] = u_dot - (2 * x * la - y * force)
            R[3] = v_dot - (2 * y * la + x * force)

            return R

        def true_sol(t):
            y = np.array(
                [
                    np.cos(phi(t)),
                    np.sin(phi(t)),
                    -np.sin(phi(t)) * phi_p(t),
                    np.cos(phi(t)) * phi_p(t),
                ]
            )

            yp = np.array(
                [
                    -np.sin(phi(t)) * phi_p(t),
                    np.cos(phi(t)) * phi_p(t),
                    -np.cos(phi(t)) * phi_p(t) ** 2 - np.sin(phi(t)) * phi_pp(t),
                    -np.sin(phi(t)) * phi_p(t) ** 2 + np.cos(phi(t)) * phi_pp(t),
                ]
            )

            return y, yp

    case v if v in [1, 2, 3]:

        def F(t, vy, vyp):
            x, y, u, v, la = vy
            x_dot, y_dot, u_dot, v_dot, _ = vyp

            force = phi_pp(t)

            R = np.zeros(5, dtype=np.common_type(vy, vyp))

            match INDEX:
                case 1:
                    R[4] = 2 * (x * u_dot + u**2 + y * v_dot + v**2)

                case 2:
                    R[4] = 2 * (x * u + y * v)

                case 3:
                    R[4] = x**2 + y**2 - 1

            R[0] = x_dot - u
            R[1] = y_dot - v
            R[2] = u_dot - (2 * x * la - y * force)
            R[3] = v_dot - (2 * y * la + x * force)

            return R

        def true_sol(t):
            y = np.array(
                [
                    np.cos(phi(t)),
                    np.sin(phi(t)),
                    -np.sin(phi(t)) * phi_p(t),
                    np.cos(phi(t)) * phi_p(t),
                    -phi_p(t) ** 2 / 2,
                ]
            )

            yp = np.array(
                [
                    -np.sin(phi(t)) * phi_p(t),
                    np.cos(phi(t)) * phi_p(t),
                    -np.cos(phi(t)) * phi_p(t) ** 2 - np.sin(phi(t)) * phi_pp(t),
                    -np.sin(phi(t)) * phi_p(t) ** 2 + np.cos(phi(t)) * phi_pp(t),
                    -phi_pp(t),
                ]
            )

            return y, yp

    case "GGL":

        def F(t, vy, vyp):
            x, y, u, v, la, mu = vy
            x_dot, y_dot, u_dot, v_dot, _, _ = vyp

            force = phi_pp(t)

            R = np.zeros(6, dtype=np.common_type(vy, vyp))
            R[0] = x_dot - (u + 2 * x * mu)
            R[1] = y_dot - (v + 2 * y * mu)
            R[2] = u_dot - (2 * x * la - y * force)
            R[3] = v_dot - (2 * y * la + x * force)
            R[4] = 2 * (x * u + y * v)
            R[5] = x**2 + y**2 - 1

            return R

        def true_sol(t):
            y = np.array(
                [
                    np.cos(phi(t)),
                    np.sin(phi(t)),
                    -np.sin(phi(t)) * phi_p(t),
                    np.cos(phi(t)) * phi_p(t),
                    -phi_p(t) ** 2 / 2,
                    np.zeros_like(t),
                ]
            )

            yp = np.array(
                [
                    -np.sin(phi(t)) * phi_p(t),
                    np.cos(phi(t)) * phi_p(t),
                    -np.cos(phi(t)) * phi_p(t) ** 2 - np.sin(phi(t)) * phi_pp(t),
                    -np.sin(phi(t)) * phi_p(t) ** 2 + np.cos(phi(t)) * phi_pp(t),
                    -phi_pp(t),
                    np.zeros_like(t),
                ]
            )

            return y, yp

    case "Hiller":

        def F(t, vy, vyp):
            x, y, u, v, _, _ = vy
            x_dot, y_dot, u_dot, v_dot, Lap, Mup = vyp

            force = phi_pp(t)

            R = np.zeros(6, dtype=np.common_type(vy, vyp))
            R[0] = x_dot - (u + 2 * x * Mup)
            R[1] = y_dot - (v + 2 * y * Mup)
            R[2] = u_dot - (2 * x * Lap - y * force)
            R[3] = v_dot - (2 * y * Lap + x * force)
            R[4] = 2 * (x * u + y * v)
            R[5] = x**2 + y**2 - 1

            return R

        def true_sol(t):
            y = np.array(
                [
                    np.cos(phi(t)),
                    np.sin(phi(t)),
                    -np.sin(phi(t)) * phi_p(t),
                    np.cos(phi(t)) * phi_p(t),
                    -PHI(t) / 2,
                    np.zeros_like(t),
                ]
            )

            yp = np.array(
                [
                    -np.sin(phi(t)) * phi_p(t),
                    np.cos(phi(t)) * phi_p(t),
                    -np.cos(phi(t)) * phi_p(t) ** 2 - np.sin(phi(t)) * phi_pp(t),
                    -np.sin(phi(t)) * phi_p(t) ** 2 + np.cos(phi(t)) * phi_pp(t),
                    -phi_p(t) ** 2 / 2,
                    np.zeros_like(t),
                ]
            )

            return y, yp


index = INDEX if INDEX in [0, 1, 2, 3] else (2 if INDEX == "GGL" else 1)

problem = DAEProblem(
    name="Circular motion",
    F=F,
    # t_span=(0.1 * np.pi, 0.45 * np.pi),
    t_span=(0, 2 * np.pi),
    index=index,
    true_sol=true_sol,
)
