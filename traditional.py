from scipy.interpolate import CubicSpline
import numpy as np
import time
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Set random seed for reproducibility
np.random.seed(0)


def poisson_fd(mesh_points: int, f_func: callable, bc: list, interval: list) -> tuple:
    """
    Solves the Poisson equation using finite differences.
    The Poisson equation is: -u''(x) = f(x), u(0) = bc[0], u(2) = bc[1].

    Parameters:
    - mesh_points: number of points in the mesh.
    - f_func: function f(x) on the right-hand side of the equation.
    - bc: boundary conditions [u(0), u(2)].
    - interval: interval [a, b] in which to solve the equation.

    Returns:
    - x: mesh points.
    - u: approximate solution at the points x.
    """

    a, b = interval
    h = (b - a) / (mesh_points - 1)
    x = np.linspace(a, b, mesh_points)

    # Construct the matrix and the vector of independent terms
    # A is the matrix of coefficients, b_vec is the vector of independent terms
    A = np.zeros((mesh_points, mesh_points))
    b_vec = np.zeros(mesh_points)

    # Fill the matrix A
    for i in range(1, mesh_points - 1):
        A[i, i - 1] = 1 / h**2
        A[i, i] = -2 / h**2
        A[i, i + 1] = 1 / h**2
        b_vec[i] = f_func(x[i])

    # Apply Dirichlet boundary conditions u(0) = bc[0] and u(2) = bc[1]
    # A[0, 0] = 1 since we are fixing u(0) = bc[0]
    # A[-1, -1] = 1 since we are fixing u(2) = bc[1]
    A[0, 0] = 1
    A[-1, -1] = 1
    b_vec[0] = bc[0]
    b_vec[-1] = bc[1]

    # Solve the linear system
    u = np.linalg.solve(A, b_vec)

    # Plot the solution
    plt.plot(x, u, label="Numerical Solution")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.title("Solution of the Poisson equation with finite differences")
    plt.legend()
    plt.show()

    return x, u


def poisson_fem(mesh_points: int, f_func: callable, bc: list, interval: list) -> tuple:
    a, b = interval
    h = (b - a) / (mesh_points - 1)
    x = np.linspace(a, b, mesh_points)

    # Stiffness matrix and load vector
    A = np.zeros((mesh_points, mesh_points))
    b_vec = np.zeros(mesh_points)

    # Build stiffness matrix and load vector
    for i in range(1, mesh_points - 1):
        # Left element
        A[i, i - 1] += 1 / h
        A[i, i] += -1 / h

        # Right element
        A[i, i] += -1 / h
        A[i, i + 1] += 1 / h

        # Load vector with Simpson's rule
        b_vec[i] = (
            (h / 6) * f_func(x[i - 1])
            + (4 * h / 6) * f_func(x[i])
            + (h / 6) * f_func(x[i + 1])
        )

    # Apply essential boundary conditions (Dirichlet)
    A[0, :] = 0
    A[0, 0] = 1
    A[-1, :] = 0
    A[-1, -1] = 1
    b_vec[0] = bc[0]
    b_vec[-1] = bc[1]

    # Solve the system
    u = np.linalg.solve(A, b_vec)

    # Plot the solution
    plt.plot(x, u, label="Numerical Solution")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.title("Solution of the Poisson equation with finite elements")
    plt.legend()
    plt.show()

    return x, u


def laplace_fd(mesh_points: int, bc: list, interval: list) -> tuple:
    """
    Solves the Laplace equation using finite differences.

    Parameters:
    - mesh_points: number of points in the mesh.
    - bc: boundary conditions [u(0), u(2)].
    - interval: interval [a, b] in which to solve the equation.

    Returns:
    - x: mesh points.
    - u: approximate solution at the points x.
    """

    a, b = interval
    h = (b - a) / (mesh_points - 1)
    x = np.linspace(a, b, mesh_points)

    # Construct the matrix and the vector of independent terms
    A = np.zeros((mesh_points, mesh_points))
    b_vec = np.zeros(mesh_points)  # f(x) = 0 for the Laplace equation

    # Fill the matrix A
    for i in range(1, mesh_points - 1):
        A[i, i - 1] = 1 / h**2
        A[i, i] = -2 / h**2
        A[i, i + 1] = 1 / h**2

    # Apply boundary conditions
    A[0, 0] = 1
    A[-1, -1] = 1
    b_vec[0] = bc[0]
    b_vec[-1] = bc[1]

    # Solve the linear system
    u = np.linalg.solve(A, b_vec)

    # Plot the solution
    plt.plot(x, u, label="Numerical Solution")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.title("Solution of the Laplace equation with finite differences")
    plt.legend()
    plt.show()

    return x, u


def laplace_fem(mesh_points: int, bc: list, interval: list) -> tuple:
    """
    Solves the Laplace equation using finite elements.

    Parameters:
    - mesh_points: number of points in the mesh.
    - bc: boundary conditions [u(0), u(2)].
    - interval: interval [a, b] in which to solve the equation.

    Returns:
    - x: mesh points.
    - u: approximate solution at the points x.
    """
    a, b = interval
    h = (b - a) / (mesh_points - 1)
    x = np.linspace(a, b, mesh_points)

    # Stiffness matrix and load vector
    A = np.zeros((mesh_points, mesh_points))
    b_vec = np.zeros(mesh_points)

    # Build stiffness matrix A and load vector b
    for i in range(1, mesh_points - 1):
        A[i, i - 1] += 1 / h
        A[i, i] += -2 / h
        A[i, i + 1] += 1 / h

    # Apply essential boundary conditions
    A[0, 0] = 1
    A[-1, -1] = 1
    b_vec[0] = bc[0]
    b_vec[-1] = bc[1]

    # Solve the system
    u = np.linalg.solve(A, b_vec)

    # Plot the solution
    plt.plot(x, u, label="Numerical Solution")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.title("Solution of the Laplace equation with finite elements")
    plt.legend()
    plt.show()

    return x, u


def heat_equation_fd(
    mesh_points: int,
    time_steps: int,
    f_func: callable,
    bc: list,
    interval: list,
    u_initial: callable,
    dt: float,
    t_max: float,
) -> tuple:
    """
    Solves the heat equation using explicit finite differences.

    Parameters:
    - mesh_points: number of points in the spatial mesh.
    - time_steps: number of time steps.
    - f_func: source function f(x, t).
    - bc: boundary conditions [u(0, t), u(2, t)].
    - interval: interval [a, b] in which to solve the equation.
    - u_initial: initial condition function u(x, 0).
    - dt: time step.
    - t_max: maximum time.

    Returns:
    - x: mesh points.
    - u: approximate solution at the points x and times t.
    """
    a, b = interval
    x = np.linspace(a, b, mesh_points)
    t = np.linspace(0, t_max, time_steps)
    h = (b - a) / (mesh_points - 1)
    r = dt / h**2

    # Check stability with the CFL condition
    # The CFL stansds for Courant-Friedrichs-Lewy condition and it states that the time
    # step dt must be small enough to ensure stability of the explicit method. For the
    # heat equation, this condition is r <= 0.5.
    if r > 0.5:
        raise ValueError(
            "The time step is too large for the stability of the explicit method."
        )

    u = np.zeros((time_steps, mesh_points))
    u[0, :] = u_initial(x)

    for n in range(len(t) - 1):
        # Apply boundary conditions
        u[n + 1, 0] = bc[0]
        u[n + 1, -1] = bc[1]
        # Update the interior
        for i in range(1, mesh_points - 1):
            u[n + 1, i] = (
                u[n, i]
                + r * (u[n, i - 1] - 2 * u[n, i] + u[n, i + 1])
                + dt * f_func(x[i], n * dt)
            )

    # Plot the solution at the final time
    # plt.plot(x, u[-1, :], label=f"Solution at t={t_max}")
    # plt.xlabel("x")
    # plt.ylabel("u(x, t)")
    # plt.title("Solution of the heat equation with finite differences")
    # plt.legend()
    # plt.show()

    return x, t, u


def compute_load_vector(
    f_func: callable, x: np.ndarray, t: float, mesh_points: int, bc: list
) -> np.ndarray:
    """
    Computes the load vector F for the element [x_i, x_{i+1}] for the heat equation.
    Parameters:
    - f_func: source function f(x, t).
    - x: spatial mesh points.
    - t: time.
    - mesh_points: number of spatial mesh points.
    - bc: boundary conditions [u(0, t), u(2, t)].
    Returns:
    - F: load vector F for the element [x_i, x_{i+1}].
    """

    F = np.zeros(mesh_points)
    F[0] = bc[0]
    F[-1] = bc[1]
    h = x[1] - x[0]  # Step size in space
    # Compute the load vector F using the trapezoidal rule

    for i in range(1, mesh_points - 1):
        # Quadrature in the element [x_i, x_{i+1}]
        F[i] += (h / 2) * (f_func(x[i], t) + f_func(x[i + 1], t))

    return F


def heat_equation_fem(
    mesh_points: int,
    time_steps: int,
    f_func: callable,
    bc: list,
    interval: list,
    u_initial: callable,
    dt: float,
    t_max: float,
) -> tuple:
    """
    Solves the heat equation using the finite element method in space and finite differences in time.

    Parameters:
    - mesh_points: number of spatial mesh points.
    - time_steps: number of time steps.
    - f_func: source function f(x, t).
    - bc: boundary conditions [u(0, t), u(2, t)].
    - interval: interval [a, b] in which to solve the equation.
    - u_initial: initial condition function u(x, 0).
    - dt: time step.
    - t_max: maximum time.

    Returns:
    - x: spatial mesh points.
    - t: time points.
    - u: approximate solution at points x and times t.
    """

    a, b = interval
    x = np.linspace(a, b, mesh_points)
    t = np.linspace(0, t_max, time_steps)
    h = (b - a) / (mesh_points - 1)

    # Mass and stiffness matrices
    # M: is the mass matrix, which comes from the temporal term of the heat equation.
    # K: is the stiffness matrix, which comes from the spatial term involving second-order derivatives.
    M = np.zeros((mesh_points, mesh_points))
    K = np.zeros((mesh_points, mesh_points))

    # Build mass and stiffness matrices
    for i in range(1, mesh_points - 1):
        # Mass
        M[i, i - 1] += h / 6
        M[i, i] += 2 * h / 3
        M[i, i + 1] += h / 6

        # Stiffness
        K[i, i - 1] += -1 / h
        K[i, i] += 2 / h
        K[i, i + 1] += -1 / h

    # Apply essential boundary conditions
    M[0, :] = 0
    M[0, 0] = 1
    M[-1, :] = 0
    M[-1, -1] = 1

    K[0, :] = 0
    K[0, 0] = 1
    K[-1, :] = 0
    K[-1, -1] = 1

    # Initial vector
    u = np.zeros((time_steps, mesh_points))
    u[0, :] = u_initial(x)

    # System matrix for explicit scheme: M u^{n+1} = (M - dt K) u^{n} + dt F
    # For greater stability, we can use implicit scheme: (M + dt K) u^{n+1} = M u^{n} + dt F (This is the same as A u^{n+1} = b)

    A = M + dt * K  # System matrix for the implicit scheme

    for n in range(0, len(t) - 1):
        ti = (n + 1) * dt

        # Load vector
        F = compute_load_vector(f_func, x, ti, mesh_points, bc)

        # Right-hand side
        b = M @ u[n, :] + dt * F

        # Apply boundary conditions
        b[0] = bc[0]
        b[-1] = bc[1]

        # Solve the system
        u[n + 1, :] = np.linalg.solve(A, b)

    # Plot the solution at the final time
    # plt.plot(x, u[-1, :], label=f"Solution at t={t_max}")
    # plt.xlabel("x")
    # plt.ylabel("u(x, t)")
    # plt.title("Solution of the heat equation with finite elements")
    # plt.legend()
    # plt.show()

    return x, t, u


def matplotlib_draw(u: np.ndarray, x: np.ndarray, t: np.ndarray):
    """
    Draw the solution of the heat equation using Matplotlib.

    Args:
        u: Solution array of shape (time_steps, mesh_points).
        x: Spatial mesh points.
        t: Time points.
    """

    fig, ax = plt.subplots(figsize=(12, 4))
    mappable = ax.imshow(
        u,
        aspect="auto",
        extent=[x[0], x[-1], t[0], t[-1]],
        origin="lower",
        cmap="inferno",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_title("Evolución de u(x,t)")
    plt.colorbar(mappable, ax=ax, label="u(x,t)")  # Add color bar
    plt.show()


def plotly_draw(u: np.ndarray, x: np.ndarray, t: np.ndarray):
    """
    Draw the solution of the heat equation using Plotly.

    Args:
        u: Solution array of shape (time_steps, mesh_points).
        x: Spatial mesh points.
        t: Time points.
    """

    fig_plotly = go.Figure(data=[go.Surface(z=u, x=x, y=t, colorscale="blues")])
    fig_plotly.update_layout(
        title="Evolution of u(x,t)",
        scene=dict(xaxis_title="x", yaxis_title="t", zaxis_title="u(x,t)"),
    )

    fig_plotly.update_traces(
        hovertemplate="x: %{x:.4f}<br>" "t: %{y:.4f}<br>" "u: %{z:.4f}<extra></extra>"
    )

    fig_plotly.show()


def residual_MSE_general(
    x: np.ndarray, t: np.ndarray, u: np.ndarray, f_func: callable
) -> float:
    """
    Computes the MSE of the residual of the PDE u_t - u_xx = f over a uniform mesh
    in [a,b] × [0,T].

    Args:
        x: array of size N with spatial nodes in [a,b].
        t: array of size M with time points in [0,T].
        u: matrix (M x N) such that u[n, i] ≈ u(x_i, t_n).
        f_func: Python function taking (x, t) and returning f(x, t).

    Returns:
        MSE: scalar value of the discrete MSE of the residual.
    """

    M, N = u.shape
    h = x[1] - x[0]
    dt = t[1] - t[0]

    sum_residual2 = 0.0

    # We define the residual only at interior nodes (i=1..N-2) and times (n=0..M-2)
    for n in range(0, M - 1):
        tn = t[n]
        for i in range(1, N - 1):
            # Approximation of ∂u/∂t at (x_i, t_n) with forward Euler
            ut_approx = (u[n + 1, i] - u[n, i]) / dt

            # Approximation of ∂²u/∂x² at (x_i, t_n) with central differences
            uxx_approx = (u[n, i - 1] - 2 * u[n, i] + u[n, i + 1]) / (h * h)

            # Residual at point (x_i, t_n)
            R = ut_approx - uxx_approx - f_func(x[i], tn)

            # Accumulate its contribution to the MSE
            sum_residual2 += (R**2) * h * dt

    # MSE of the residual
    MSE = sum_residual2 / (M * N)
    return MSE


def bilinear_interpolate(
    x: np.ndarray, t: np.ndarray, u: np.ndarray, x_pt: float, t_pt: float
) -> float:
    """
    Perform bilinear interpolation to estimate the value of u at (x_pt, t_pt).

    Args:
        x: array of size N with spatial nodes
        t: array of size M with temporal nodes
        u: matrix (M x N) with values u[n, i]
        x_pt: spatial point where to interpolate
        t_pt: temporal point where to interpolate

    Returns:
        u_interp: interpolated value at (x_pt, t_pt)
    """
    # Find indices i, j such that x[i] <= x_pt <= x[i+1], t[j] <= t_pt <= t[j+1]
    i = np.searchsorted(x, x_pt) - 1
    j = np.searchsorted(t, t_pt) - 1
    i = np.clip(i, 0, len(x) - 2)
    j = np.clip(j, 0, len(t) - 2)

    x0, x1 = x[i], x[i + 1]
    t0, t1 = t[j], t[j + 1]
    u00 = u[j, i]
    u01 = u[j, i + 1]
    u10 = u[j + 1, i]
    u11 = u[j + 1, i + 1]

    # Interpolation weights
    dx = (x_pt - x0) / (x1 - x0)
    dt = (t_pt - t0) / (t1 - t0)

    # Bilinear interpolation
    u_interp = (
        u00 * (1 - dx) * (1 - dt)
        + u01 * dx * (1 - dt)
        + u10 * (1 - dx) * dt
        + u11 * dx * dt
    )
    return u_interp


def residual_MSE_offgrid(
    x: np.ndarray, t: np.ndarray, u: np.ndarray, f_func: callable, num_samples: int
) -> float:
    """
    Calculate an estimate of the MSE error of the PDE residual evaluated at random
    off-grid points, using bilinear interpolation and small finite differences
    for derivatives.

    Args:
        x: array of size N with spatial nodes
        t: array of size M with temporal nodes
        u: matrix (M x N) with numerical values u[n, i]
        f_func: function that takes (x, t) and returns f(x, t)
        num_samples: number of random points to sample inside (a, b)×(0, T)

    Returns:
        MSE_offgrid: estimated MSE of the off-grid residual
    """

    a, b = x[0], x[-1]
    T = t[-1]
    h = x[1] - x[0]
    dt = t[1] - t[0]

    # Define small offsets for finite differences
    eps_x = h / 50
    eps_t = dt / 50

    sum_R2 = 0.0

    # Sample random points in the interior domain, avoiding small margins
    xs = np.random.uniform(a + eps_x, b - eps_x, num_samples)
    ts = np.random.uniform(eps_t, T - eps_t, num_samples)

    for x_pt, t_pt in zip(xs, ts):
        # Interpolate u at the point (x_pt, t_pt)
        u_center = bilinear_interpolate(x, t, u, x_pt, t_pt)

        # Estimate ∂u/∂t using central difference in time
        u_t_plus = bilinear_interpolate(x, t, u, x_pt, t_pt + eps_t)
        u_t_minus = bilinear_interpolate(x, t, u, x_pt, t_pt - eps_t)
        u_t_approx = (u_t_plus - u_t_minus) / (2 * eps_t)

        # Estimate ∂²u/∂x² using central difference in space
        u_x_plus = bilinear_interpolate(x, t, u, x_pt + eps_x, t_pt)
        u_x_minus = bilinear_interpolate(x, t, u, x_pt - eps_x, t_pt)
        u_xx_approx = (u_x_plus - 2 * u_center + u_x_minus) / (eps_x * eps_x)

        # Calculate the residual at (x_pt, t_pt)
        R_pt = u_t_approx - u_xx_approx - f_func(x_pt, t_pt)
        if abs(R_pt) < 1:  # Filter beacuse problems with second derivate
            sum_R2 += R_pt**2

    # Approximate the MSE
    MSE_offgrid = sum_R2 / num_samples
    return MSE_offgrid


def build_row_splines(x: np.ndarray, t: np.ndarray, u: np.ndarray) -> list:
    """
    Precompute 1D cubic splines in x for each fixed time t[n].

    Args:
        x: array of shape (N,) with spatial nodes
        t: array of shape (M,) with temporal nodes
        u: array of shape (M, N) with u[n, i] = u(t[n], x[i])

    Returns:
        splines_x: list of length M, where splines_x[n] is a CubicSpline over x for u[n, :].
    """

    M, N = u.shape
    splines_x = [CubicSpline(x, u[n, :], bc_type="natural") for n in range(M)]
    return splines_x


def residual_MSE_two_stage(
    x: np.ndarray, t: np.ndarray, u: np.ndarray, f_func: callable, num_samples: int
) -> float:
    """
    Estimate the PDE-residual mean squared error (MSE) by sampling random off-grid points
    and using two-stage 1D cubic splines in x and t.

    Args:
        x: array of shape (N,) with spatial nodes
        t: array of shape (M,) with temporal nodes
        u: array of shape (M, N) with u[n, i] = u(t[n], x[i])
        f_func: function f(x, t) -> f(x, t)
        num_samples: number of random points to sample inside (a, b)×(0, T)

    Returns:
        mse_offgrid: mean squared residual = (1/num_samples) * sum [R(x*,t*)]^2
    """

    M, N = u.shape
    a, b = x[0], x[-1]
    T = t[-1]

    # Build row splines in x for each t[n]
    splines_x = build_row_splines(x, t, u)

    sum_R2 = 0.0
    # Sample uniform interior points
    xs = np.random.uniform(a, b, num_samples)
    ts = np.random.uniform(0, T, num_samples)

    for x_pt, t_pt in zip(xs, ts):
        # 1) Compute alpha_n = d^2/dx^2 S_n(x_pt) for each row spline
        alpha = np.array([splines_x[n].derivative(nu=2)(x_pt) for n in range(M)])
        # Build cubic spline in t for alpha
        spline_alpha_t = CubicSpline(t, alpha, bc_type="natural")
        u_xx_val = spline_alpha_t(t_pt)  # ∂²u/∂x² at (x_pt, t_pt)

        # 2) Compute vals_at_x[n] = S_n(x_pt) for each row
        vals_at_x = np.array([splines_x[n](x_pt) for n in range(M)])
        # Build cubic spline in t for u-values at x_pt
        spline_vals_t = CubicSpline(t, vals_at_x, bc_type="natural")
        u_t_val = spline_vals_t.derivative(nu=1)(t_pt)  # ∂u/∂t at (x_pt, t_pt)

        # Compute residual R = u_t - u_xx - f
        f_val = f_func(x_pt, t_pt)
        R_pt = u_t_val - u_xx_val - f_val
        sum_R2 += R_pt**2

    mse_offgrid = sum_R2 / num_samples
    return mse_offgrid


if __name__ == "__main__":
    f = lambda x: np.sin(np.pi * x)
    interval = [0, 4]
    boundary_conditions = [0, 0]
    mesh_points = 100
    # x, u = poisson_fem(mesh_points, f, boundary_conditions, interval)
    # x, u = poisson_fd(mesh_points, f, boundary_conditions, interval)

    interval = [0, 4]
    boundary_conditions = [-1, 0]
    mesh_points = 100
    # x, u = laplace_fd(mesh_points, boundary_conditions, interval)
    # x, u = laplace_fem(mesh_points, boundary_conditions, interval)

    # Example functions
    f = lambda x, t: 0  # Without source term
    x0 = 1.5  # Center of the Gaussian source term
    f = lambda x, t: np.exp(-((x - x0) ** 2)) * np.sin(t)  # Gaussian source term

    # Example usage
    interval = [0, 4]
    dt = 0.0001
    t_max = 2
    time_steps = int(t_max / dt) + 1
    boundary_conditions = [0, 0]
    mesh_points = 400
    f = lambda x, t: np.sin(np.pi * x / 2) * np.exp(-10 * t)  # Source term
    u0 = lambda x: np.sin(np.pi * x / 2)  # Temperature distribution at t=0
    method = "fd"  # Choose between 'fem' or 'fd'

    t0_trad = time.time()
    if method == "fem":
        # Solve using finite element method
        x, t, u = heat_equation_fem(
            mesh_points, time_steps, f, boundary_conditions, interval, u0, dt, t_max
        )
    elif method == "fd":
        # Solve using finite difference method
        x, t, u = heat_equation_fd(
            mesh_points, time_steps, f, boundary_conditions, interval, u0, dt, t_max
        )
    else:
        raise ValueError("Method must be 'fem' or 'fd'.")
    t1_trad = time.time()
    print(
        f"Time taken for traditional method ({method}): {t1_trad - t0_trad:.4f} seconds"
    )

    num_samples = 400
    # Compute the MSE of the residual (not really useful)
    MSE_residual = residual_MSE_general(x, t, u, f)
    print(f"On-grid MSE of the residual: {MSE_residual:.8f}")

    # Calculate off-grid MSE of the residual (weak due to second derivative):
    MSE_offgrid = residual_MSE_offgrid(x, t, u, f, num_samples)
    print(f"Off-grid MSE of the residual: {MSE_offgrid:.8f}")

    # Calculate the mean squared error (MSE) of the off-grid residual using two-stage splines
    MSE_offgrid_splines = residual_MSE_two_stage(x, t, u, f, num_samples)
    print(f"Off-grid MSE with cubic splines: {MSE_offgrid_splines:.8e}")

    # Draw the solution
    # matplotlib_draw(u, x, t)
    # plotly_draw(u, x, t)
