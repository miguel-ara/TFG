from pinn import HeatEquationPINN, load_data
from traditional import heat_equation_fd, heat_equation_fem, residual_MSE_two_stage
import torch
import numpy as np
import plotly.graph_objects as go
import time

# Set random seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)


def draw_solution_combined(
    x: np.ndarray,
    t: np.ndarray,
    u: np.ndarray,
    X_train: np.ndarray,
    T_train: np.ndarray,
    u_pred_train: np.ndarray,
    x_ic: np.ndarray,
    t_bc: np.ndarray,
    mesh_points: int,
    time_steps: int,
    u_pred: np.ndarray,
    bc_points: int,
    ic_points: int,
    method: str,
):
    """
    Plots the combined solution of the heat equation.

    Args:
        x: Points on the x-axis.
        t: Points on the t-axis.
        u: Solution at the points (x, t).
        X_train: Training points in x.
        T_train: Training points in t.
        u_pred_train: Predicted solution at the training points.
        x_ic: Initial condition points in x.
        t_bc: Boundary condition points in t.
        mesh_points: Number of points in the mesh.
        time_steps: Number of time steps.
        u_pred: Predicted solution at all points.
        bc_points: Number of boundary condition points multiplier for time_steps.
        ic_points: Number of initial condition points multiplier for mesh_points.
        method: Name of the traditional method used.
    """

    # Flatten the arrays for the scatter plot
    u_pred_T = u_pred.T.flatten()
    u_pred_train = u_pred_train.flatten()
    x_ic = np.tile(x_ic, time_steps * bc_points)  # Repeat x_np for each value of t
    t_bc = np.repeat(t_bc, mesh_points * ic_points)  # Repeat t_np for each value of x
    fig_plotly = go.Figure()

    # 1) The semi‑transparent surface
    fig_plotly.add_trace(
        go.Surface(
            x=x,
            y=t,
            z=u,
            colorscale="Blues",
            opacity=0.5,
            name=f"Traditional ({method})",
            showlegend=True,
        )
    )

    # 2) All training points of the PINN
    fig_plotly.add_trace(
        go.Scatter3d(
            x=X_train,
            y=T_train,
            z=u_pred_train,
            mode="markers",
            marker=dict(size=1, color="red", opacity=0.8),
            name="Training points of the PINN",
            showlegend=True,
        )
    )

    # 3) Sub‑sample along t=0
    fig_plotly.add_trace(
        go.Scatter3d(
            x=x_ic[: mesh_points * ic_points],
            y=t_bc[: mesh_points * ic_points],
            z=u_pred_T[: mesh_points * ic_points],
            mode="markers",
            marker=dict(size=3, color="blue", opacity=0.8),
            name="Condición inicial",
            showlegend=True,
        )
    )

    # 4) Sub‑sample along x_min and x_max
    slice_min = np.arange(0, x_ic.shape[0], mesh_points * ic_points)
    slice_max = np.arange(
        mesh_points * ic_points - 1, x_ic.shape[0], mesh_points * ic_points
    )
    both_slices = np.concatenate([slice_min, slice_max])

    # Now draw them in one trace:
    fig_plotly.add_trace(
        go.Scatter3d(
            x=x_ic[both_slices],
            y=t_bc[both_slices],
            z=u_pred_T[both_slices],
            mode="markers",
            marker=dict(size=3, color="green", opacity=0.8),
            name="Boundary conditions",
        )
    )

    fig_plotly.update_layout(
        title="Evolution of u(x,t)",
        scene=dict(xaxis_title="x", yaxis_title="t", zaxis_title="u(x,t)"),
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor="rgba(255,255,255,0.7)",  # semi‑transparent white background
        ),
    )

    fig_plotly.update_traces(
        hovertemplate="x: %{x:.4f}<br>" "t: %{y:.4f}<br>" "u: %{z:.4f}<extra></extra>"
    )

    fig_plotly.show()


def draw_only_nets(
    x: np.ndarray,
    t: np.ndarray,
    u: np.ndarray,
    X_pred: np.ndarray,
    T_pred: np.ndarray,
    u_pred: np.ndarray,
    method: str,
    scatter: bool = False,
):
    """
    Draw the neural network predictions alongside the traditional method solution.

    Args:
        x: Spatial points.
        t: Temporal points.
        u: Solution at the points (x, t).
        X_pred: Predicted spatial points.
        T_pred: Predicted temporal points.
        u_pred: Predicted solution at the points (X_pred, T_pred).
        method: Name of the traditional method used.
        scatter: Whether to use scatter plot or mesh plot. Defaults to False.
    """

    # Flatten the arrays for the scatter plot
    u_pred = u_pred.flatten()

    fig_plotly = go.Figure()

    # 1) The semi‑transparent surface with the traditional method
    fig_plotly.add_trace(
        go.Surface(
            x=x,
            y=t,
            z=u,
            colorscale="Blues",
            opacity=0.8,
            name=f"Traditional ({method})",
            showlegend=True,
        )
    )

    if scatter:
        # 2) All predicted points of the PINN as a scatter plot
        fig_plotly.add_trace(
            go.Scatter3d(
                x=X_pred,
                y=T_pred,
                z=u_pred,
                mode="markers",
                marker=dict(size=3, color="green", opacity=0.8),
                name="PINN",
                showlegend=True,
            )
        )

    else:
        # 2) All predicted points of the PINN as a surface
        fig_plotly.add_trace(
            go.Mesh3d(
                x=X_pred,
                y=T_pred,
                z=u_pred,
                colorscale="Reds",
                opacity=0.5,
                name="PINN",
                showlegend=True,
            )
        )

    fig_plotly.update_layout(
        title="Evolución de u(x,t)",
        scene=dict(xaxis_title="x", yaxis_title="t", zaxis_title="u(x,t)"),
        legend=dict(
            x=0.2,
            y=0.98,
            bgcolor="rgba(255,255,255,0.7)",  # Semi‑transparent white background
        ),
    )

    fig_plotly.update_traces(
        hovertemplate="x: %{x:.4f}<br>" "t: %{y:.4f}<br>" "u: %{z:.4f}<extra></extra>"
    )

    fig_plotly.show()


if __name__ == "__main__":
    # ---------------- Problem Definition ----------------
    boundary_conditions = [0, 0]  # Boundary conditions at x=0 and x=L
    interval = [0, 4]
    t_max = 2

    # PINN
    dt = 0.1
    time_steps_pinn = int(t_max / dt) + 1
    mesh_points_pinn = 41
    data_percentage = 1  # Percentage of data to use for training
    reduced_mesh_points = round(mesh_points_pinn * data_percentage)
    bc_points = 1  # Multiplier of mesh_points to have more points at the boundary
    ic_points = 1  # Multiplier of mesh_points to have more points at the initial cond

    u0 = lambda x: torch.sin(np.pi * x / 2)  # Initial condition
    f = lambda x, t: torch.sin(np.pi * x / 2) * torch.exp(-10 * t)  # Source term

    mode = "random"  # "uniform" or "random"

    X_train, T_train, x_ic, t_bc = load_data(
        interval,
        mesh_points_pinn,
        time_steps_pinn,
        reduced_mesh_points,
        t_max,
        bc_points,
        ic_points,
        mode,
    )
    train_data = [X_train, T_train, x_ic, t_bc]

    # ---------------- Training ----------------
    layers = [2, 10, 10, 1]
    lr = 1e-2
    epochs = 2000
    print_every = 100
    weight_factors = [1, 1, 1]  # [equation, initial condition, boundary conditions]
    pinn = HeatEquationPINN(layers, lr)

    t0_pinn = time.time()
    pinn.train(
        epochs,
        print_every,
        train_data,
        weight_factors,
        boundary_conditions,
        u0,
        f,
        interval,
        mode,
    )
    t1_pinn = time.time()
    print(f"Training time for PINN: {t1_pinn - t0_pinn:.4f} seconds")

    # Define where we want the network to PREDICT
    X, T = torch.meshgrid(x_ic.squeeze(), t_bc.squeeze(), indexing="ij")
    X_pred, T_pred = X.flatten().unsqueeze(1), T.flatten().unsqueeze(1)

    # Prediction of the original training points
    u_pred_train = pinn.predict(X_train, T_train).reshape(
        reduced_mesh_points, time_steps_pinn
    )

    # Prediction of the new uniformly distributed points
    u_pred = pinn.predict(X_pred, T_pred).reshape(
        mesh_points_pinn * ic_points, time_steps_pinn * bc_points
    )

    # ---------------- Plot ----------------
    x_ic = x_ic.detach().numpy().flatten()
    t_bc = t_bc.detach().numpy().flatten()
    X_train = X_train.detach().numpy().flatten()
    T_train = T_train.detach().numpy().flatten()
    X_pred = X_pred.detach().numpy().flatten()
    T_pred = T_pred.detach().numpy().flatten()

    # Traditional method
    dt = 0.0001
    time_steps = int(t_max / dt) + 1
    mesh_points = 101
    f = lambda x, t: np.sin(np.pi * x / 2) * np.exp(-10 * t)  # Source term
    u0 = lambda x: np.sin(np.pi * x / 2)  # Initial condition
    method = "fem"  # "fd" or "fem"

    t0_trad = time.time()
    if method == "fem":
        x, t, u = heat_equation_fem(
            mesh_points, time_steps, f, boundary_conditions, interval, u0, dt, t_max
        )
    elif method == "fd":
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
    MSE_offgrid_splines = residual_MSE_two_stage(x, t, u, f, num_samples)
    print(f"Off-grid MSE with cubic splines with traditional method ({method}): {MSE_offgrid_splines:.8f}")

    x_tensor = torch.tensor(x, dtype=torch.float32)
    t_tensor = torch.tensor(t, dtype=torch.float32)
    X, T = torch.meshgrid(x_tensor, t_tensor, indexing="ij")
    X_new, T_new = X.flatten().unsqueeze(1), T.flatten().unsqueeze(1)
    prediction = (
        pinn.predict(X_new, T_new).reshape(x_tensor.shape[0], t_tensor.shape[0]).T
    )
    MSE_offgrid_splines_pinn = residual_MSE_two_stage(x, t, prediction, f, num_samples)
    print(f"Off-grid MSE with cubic splines with PINN: {MSE_offgrid_splines_pinn:.8f}")

    draw_solution_combined(
        x,
        t,
        u,
        X_train,
        T_train,
        u_pred_train,
        x_ic,
        t_bc,
        mesh_points_pinn,
        time_steps_pinn,
        u_pred,
        bc_points,
        ic_points,
        method,
    )

    draw_only_nets(x, t, u, X_pred, T_pred, u_pred, method, scatter=False)
