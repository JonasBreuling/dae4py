import vtk
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.optimize._numdiff import approx_derivative
from scipy.linalg import issymmetric
from dae4py.radau import solve_dae_radau


def create_grid(Lx, Ly, nx, ny):
    """Generate grid."""

    # number of nodes
    Nx = nx + 1
    Ny = ny + 1

    # grid size (equispaced)
    dx = Lx / nx
    dy = Ly / ny

    # coordinate of each grid (cell center)
    xij = (np.arange(nx) + 0.5) * dx
    yij = (np.arange(ny) + 0.5) * dy

    # coordinate of each grid (cell corner)
    xi2j2 = (np.arange(Nx)) * dx
    yi2j2 = (np.arange(Ny)) * dy

    return xij, yij, xi2j2, yi2j2, dx, dy


def initial_conditions(nx, ny):
    """Set initial conditions."""

    u_init = np.zeros((nx + 1, ny + 2))
    v_init = np.zeros((nx + 2, ny + 1))
    p_init = np.zeros((nx, ny))

    ut_init = np.zeros((nx + 1, ny + 2))
    vt_init = np.zeros((nx + 2, ny + 1))
    pt_init = np.zeros((nx, ny))

    y0 = np.concatenate(
        (
            u_init[1:nx, 1 : ny + 1].flatten(),
            v_init[1 : nx + 1, 1:ny].flatten(),
            p_init.flatten(),
        )
    )
    yp0 = np.concatenate(
        (
            ut_init[1:nx, 1 : ny + 1].flatten(),
            vt_init[1 : nx + 1, 1:ny].flatten(),
            pt_init.flatten(),
        )
    )

    return y0, yp0


def apply_boundary_conditions(u_red, v_red, p_red, BC):
    """Applies Dirichlet boundary conditions."""

    # unpack dictionary
    u_left = BC["u_left"]
    u_right = BC["u_right"]
    u_bot = BC["u_bot"]
    u_top = BC["u_top"]

    v_left = BC["v_left"]
    v_right = BC["v_right"]
    v_bot = BC["v_bot"]
    v_top = BC["v_top"]

    # Dirichlet boundary conditions for velocities
    if u_left is None:
        u_red[0, :] = u_red[1, :]
    else:
        u_red[0, :] = u_left

    if u_right is None:
        u_red[-1, :] = u_red[-2, :]
    else:
        u_red[-1, :] = u_right

    if u_bot is None:
        u_red[:, 0] = u_red[:, 1]
    else:
        u_red[:, 0] = 2 * u_bot - u_red[:, 1]

    if u_top is None:
        u_red[:, -1] = u_red[:, -2]
    else:
        u_red[:, -1] = 2 * u_top - u_red[:, -2]

    if v_left is None:
        v_red[0, :] = v_red[1, :]
    else:
        v_red[0, :] = 2 * v_left - v_red[1, :]

    if v_right is None:
        v_red[-1, :] = v_red[-2, :]
    else:
        v_red[-1, :] = 2 * v_right - v_red[-2, :]

    if v_top is None:
        v_red[1:-1, -1] = v_red[1:-1, -2]
    else:
        v_red[:, -1] = v_top

    if v_bot is None:
        v_red[1:-1, 0] = v_red[1:-1, 1]
    else:
        v_red[:, 0] = v_bot


def redundant_coordinates(t, y, yp, nx, ny, BC):
    """Converts state and derivatives into structured 2D arrays with boundary
    conditions applied."""
    # unpack state vector and derivatives
    nu = (nx - 1) * ny
    nv = nx * (ny - 1)
    split = np.cumsum([nu, nv])
    u, v, p = np.array_split(y, split)
    ut, vt, pt = np.array_split(yp, split)

    # reshape 2D
    u = u.reshape((nx - 1, ny))
    v = v.reshape((nx, ny - 1))
    p = p.reshape((nx, ny))
    ut = ut.reshape((nx - 1, ny))
    vt = vt.reshape((nx, ny - 1))
    pt = pt.reshape((nx, ny))

    # build redundant coordinates
    u_red = np.zeros((nx + 1, ny + 2))
    v_red = np.zeros((nx + 2, ny + 1))
    p_red = np.zeros((nx, ny))
    ut_red = np.zeros((nx + 1, ny + 2))
    vt_red = np.zeros((nx + 2, ny + 1))
    pt_red = np.zeros((nx, ny))

    # interior velocites are the unknowns; all pressures are unknown
    u_red[1:-1, 1:-1] = u
    v_red[1:-1, 1:-1] = v
    p_red = p

    ut_red[1:-1, 1:-1] = ut
    vt_red[1:-1, 1:-1] = vt
    pt_red = pt

    # boundary conditions
    apply_boundary_conditions(u_red, v_red, p_red, BC(t))

    return u_red, v_red, p_red, ut_red, vt_red, pt_red


def F(t, y, yp):
    # set boundary conditions
    u, v, p, ut, vt, pt = redundant_coordinates(t, y, yp, nx, ny, BC)
    # p = pt  # note: Index reduction!

    # interpolate velocities
    uij = 0.5 * (u[:-1, 1:-1] + u[1:, 1:-1])
    u2ij = uij**2
    vij = 0.5 * (v[1:-1, :-1] + v[1:-1, 1:])
    v2ij = vij**2
    ui2j2 = 0.5 * (u[:, :-1] + u[:, 1:])
    vi2j2 = 0.5 * (v[:-1] + v[1:])

    # momentum equation for u
    Fu = (
        ut[1:-1, 1:-1]
        + (u2ij[1:] - u2ij[:-1]) / dx
        + (ui2j2[1:-1, 1:] * vi2j2[1:-1, 1:] - ui2j2[1:-1, :-1] * vi2j2[1:-1, :-1]) / dy
        + (p[1:] - p[:-1]) / dx
        - nu
        * (
            (u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[:-2, 1:-1]) / dx**2
            + (u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, :-2]) / dy**2
        )
    )

    # momentum equation for v
    Fv = (
        vt[1:-1, 1:-1]
        + (ui2j2[1:, 1:-1] * vi2j2[1:, 1:-1] - ui2j2[:-1, 1:-1] * vi2j2[:-1, 1:-1]) / dx
        + (v2ij[:, 1:] - v2ij[:, :-1]) / dy
        + (p[:, 1:] - p[:, :-1]) / dy
        - nu
        * (
            (v[2:, 1:-1] - 2 * v[1:-1, 1:-1] + v[:-2, 1:-1]) / dx**2
            + (v[1:-1, 2:] - 2 * v[1:-1, 1:-1] + v[1:-1, :-2]) / dy**2
        )
    )

    # continuity equation
    Fp = (u[1:, 1:-1] - u[:-1, 1:-1]) / dx + (v[1:-1, 1:] - v[1:-1, :-1]) / dy

    return np.concatenate(
        (
            Fu.flatten(),
            Fv.flatten(),
            Fp.flatten(),
        )
    )


def animate(x, y, u, v, p, interval=1):
    fig, ax = plt.subplots()
    # plt.gca().invert_yaxis()

    def update(num):
        ax.clear()
        ax.set_xlim(-0.25 * Lx, 1.25 * Lx)
        ax.set_ylim(-0.25 * Ly, 1.25 * Ly)
        ax.set_aspect("equal")
        ax.plot(x, y, "ok")

        contourf = ax.contourf(
            x, y, np.sqrt(u[:, :, num] ** 2 + v[:, :, num] ** 2), alpha=0.5
        )
        streamplot = ax.streamplot(x, y, u[:, :, num], v[:, :, num], density=1.5)
        return contourf, streamplot

    anim = animation.FuncAnimation(
        fig, update, frames=u.shape[-1], interval=interval, blit=False, repeat=True
    )
    plt.show()


############
# parameters
############
# domain
h = 1
H = 2 * h
Lx = 10 * h
Ly = 2 * h + H

# number of cell centers per h
nxh, nyh = 4, 4

nx = int(Lx / h * nxh)
ny = int(Ly / h * nyh)
print(f"nx: {nx}")
print(f"ny: {ny}")

# kinematic viscosity
nu = 1

# initial conditions
y0, yp0 = initial_conditions(nx, ny)
print(f"DOF's: {len(y0)}")

# generate the grid
xij, yij, xi2j2, yi2j2, dx, dy = create_grid(Lx, Ly, nx, ny)

global BC
BC = None

def sparsity_pattern():
    global BC
    # boundary conditions (no B.C. for Jacobian sparsity pattern)
    def BC(t):
        return {
            "u_top": None,
            "u_bot": None,
            "u_left": None,
            "u_right": None,
            "v_top": None,
            "v_bot": None,
            "v_left": None,
            "v_right": None,
        }

    J0 = approx_derivative(
        lambda y: F(0, y, np.zeros_like(y)),
        y0 + np.random.rand(len(y0)) * 1e-6,  # perturbation for numerical stability
    )
    Jp0 = approx_derivative(
        lambda yp: F(0, np.zeros_like(yp), yp),
        yp0 + np.random.rand(len(yp0)) * 1e-6,  # perturbation for numerical stability,
    )

    fig, ax = plt.subplots(1, 2)
    ax[0].spy(J0, markersize=1)
    ax[0].set_title("Jacobian J")
    ax[1].spy(Jp0, markersize=1)
    ax[1].set_title("Jacobian Jp")
    plt.show()

def run():
    global BC

    # reference velocity
    U0 = 50

    # linear spaced vertical mesh points
    y = np.linspace(0, Ly, ny + 2)

    # laminar velocity profile
    u_left_profile = U0 * np.maximum(
        0, 4 * (y - h) * (Ly - 2 * h - (y - h)) / (Ly - 2 * h) ** 2
    )

    # boundary conditions
    def BC(t):
        return {
            "u_top": 0,
            "u_bot": 0,
            "u_left": u_left_profile,
            "u_right": None,
            "v_top": 0,
            "v_bot": 0,
            "v_left": 0,
            "v_right": None,
        }

    # time span
    t0 = 0
    t1 = 1
    t_span = (t0, t1)
    t_eval = np.linspace(t0, t1, num=int(5e2))

    # solver options
    atol = rtol = 1e-4

    # solve the system
    start = time.time()
    sol = solve_dae_radau(
        F,
        y0,
        yp0,
        t_span,
        atol=atol,
        rtol=rtol,
        t_eval=t_eval,
        s=5,
    )
    end = time.time()
    print(f"elapsed time: {end - start}")
    t = sol.t
    sol_y = sol.y.T
    sol_yp = sol.yp.T
    print(sol)

    # reconstruct solution
    Xij, Yij = np.meshgrid(xij, yij, indexing="ij")
    Xi2j2, Yi2j2 = np.meshgrid(xi2j2, yi2j2, indexing="ij")

    nt = len(t)
    u = np.zeros((nx + 1, ny + 2, nt))
    v = np.zeros((nx + 2, ny + 1, nt))
    p = np.zeros((nx, ny, nt))
    ut = np.zeros((nx + 1, ny + 2, nt))
    vt = np.zeros((nx + 2, ny + 1, nt))
    pt = np.zeros((nx, ny, nt))
    for i in range(nt):
        u[:, :, i], v[:, :, i], p[:, :, i], ut[:, :, i], vt[:, :, i], pt[:, :, i] = (
            redundant_coordinates(t[i], sol_y[:, i], sol_yp[:, i], nx, ny, BC)
        )

    # interpolate velocity at cell centers and cell corners
    uij = 0.5 * (u[:-1, 1:-1] + u[1:, 1:-1])
    ui2j2 = 0.5 * (u[:, :-1] + u[:, 1:])
    vij = 0.5 * (v[1:-1, :-1] + v[1:-1, 1:])
    vi2j2 = 0.5 * (v[:-1, :] + v[1:, :])

    # transpose data for "xy" meshgrid and streamplot
    # ui2j2 = ui2j2.transpose(1, 0, 2)
    # vi2j2 = vi2j2.transpose(1, 0, 2)
    uij = uij.transpose(1, 0, 2)
    vij = vij.transpose(1, 0, 2)
    p = p.transpose(1, 0, 2)

    Xij, Yij = np.meshgrid(xij, yij, indexing="xy")
    animate(Xij, Yij, uij, vij, p)

    ###############################
    # visualize reattachment length
    # xr = 6.28h due to Le & Moin
    ###############################
    fig, ax = plt.subplots()
    ax.title.set_text(f"reattachment length")
    ax.plot(Xi2j2[:, 1] / h, ui2j2[:, 1, -1], label="u_{i+1/2,-1/2}")
    ax.plot(Xi2j2[:, 1] / h, 0 * ui2j2[:, 1, -1])
    ax.grid()
    ax.legend()
    ax.set_xlabel("x / h")
    plt.show()

    ############
    # vtk export
    ############
    # define output directory
    output_dir = "backward_facing_step"
    os.makedirs(output_dir, exist_ok=True)

    # create .pvd file
    pvd_filename = os.path.join(output_dir, "backward_facing_step.pvd")
    pvd_file = open(pvd_filename, "w")
    pvd_file.write('<?xml version="1.0"?>\n')
    pvd_file.write('  <VTKFile type="Collection">\n')
    pvd_file.write("    <Collection>\n")

    # vtu writer
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetDataModeToAscii()  # debug

    for k in range(nt):
        filename = f"data_{k:03d}.vtu"
        full_path = os.path.join(output_dir, filename)

        # create grid
        grid = vtk.vtkUnstructuredGrid()

        # add points
        vtkpoints = vtk.vtkPoints()
        vtkpoints.Allocate(Xi2j2.size)
        for i in range(nx):
            for j in range(ny):
                vtkpoints.InsertNextPoint((Xi2j2[i, j], Yi2j2[i, j], 0))
                vtkpoints.InsertNextPoint((Xi2j2[i + 1, j], Yi2j2[i + 1, j], 0))
                vtkpoints.InsertNextPoint((Xi2j2[i + 1, j + 1], Yi2j2[i + 1, j + 1], 0))
                vtkpoints.InsertNextPoint((Xi2j2[i, j + 1], Yi2j2[i, j + 1], 0))

        grid.SetPoints(vtkpoints)

        # define cells
        grid.Allocate(nx * ny)
        cell_type = vtk.VTK_QUAD  # 90
        offset = 0
        connectivity = np.arange(4)
        for i in range(nx):
            for j in range(ny):
                grid.InsertNextCell(cell_type, len(connectivity), connectivity + offset)
                offset += 4

        # add velocity as point data
        u_val = ui2j2[:, :, k]
        v_val = vi2j2[:, :, k]
        pdata = grid.GetPointData()
        u_array = vtk.vtkDoubleArray()
        u_array.SetName("u")
        u_array.SetNumberOfComponents(3)
        for i in range(nx):
            for j in range(ny):
                u_array.InsertNextTuple3(u_val[i, j], v_val[i, j], 0)
                u_array.InsertNextTuple3(u_val[i + 1, j], v_val[i + 1, j], 0)
                u_array.InsertNextTuple3(u_val[i + 1, j + 1], v_val[i + 1, j + 1], 0)
                u_array.InsertNextTuple3(u_val[i, j + 1], v_val[i, j + 1], 0)
        pdata.AddArray(u_array)

        # add pressure as cell data
        p_val = pt[:, :, k]
        cdata = grid.GetCellData()
        carray = vtk.vtkDoubleArray()
        carray.SetName("p")
        carray.SetNumberOfComponents(1)
        for i in range(nx):
            for j in range(ny):
                carray.InsertNextTuple1(p_val[i, j])
        cdata.AddArray(carray)

        # write the VTU file
        writer.SetFileName(full_path)
        writer.SetInputData(grid)
        writer.Write()

        # add to PVD file
        pvd_file.write(f'     <DataSet timestep="{t[k]}" file="{filename}"/>\n')

    # finalize pvd file
    pvd_file.write("    </Collection>\n")
    pvd_file.write("  </VTKFile>\n")
    pvd_file.close()

if __name__ == "__main__":
    sparsity_pattern()
    # run()
