import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from traditional import residual_MSE_two_stage

# Set random seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)


# ---------------- NN Architecture ----------------
class PINN(nn.Module):
    def __init__(self, layers: list):
        """
        Initialize the PINN model.

        Args:
            layers: List of integers representing the number of neurons in each layer.
        """
        super(PINN, self).__init__()
        self.net = nn.Sequential()
        for i in range(len(layers) - 1):
            self.net.add_module(f"layer_{i}", nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                self.net.add_module(f"activation_{i}", nn.SiLU())

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        Forward pass for the PINN.
        Combines spatial and temporal inputs and passes them through the network.

        Args:
            x: Spatial input tensor.
            t: Temporal input tensor.

        Returns:
            Output tensor after passing through the network.
        """
        inputs = torch.cat([x, t], dim=1)
        return self.net(inputs)


class HeatEquationPINN:
    def __init__(self, layers: list, lr: float = 1e-2):
        """
        Initialize the HeatEquationPINN.
        Args:
            layers: List of integers representing the number of neurons in each layer.
            lr: Learning rate for the optimizer.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PINN(layers).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_values = {
            "heat_eq_loss": [],
            "initial_condition_loss": [],
            "boundary_condition_loss": [],
            "total_loss": [],
        }

    def heat_equation_loss(
        self, X_flat: torch.Tensor, T_flat: torch.Tensor, f: callable
    ) -> torch.Tensor:
        """
        Compute the loss for the heat equation PDE.
        Args:
            X_flat: Flattened spatial input tensor.
            T_flat: Flattened temporal input tensor.
            f: Function representing the source term in the PDE.
        Returns:
            Mean squared error loss for the heat equation.
        """
        u = self.model(X_flat, T_flat)
        u_t = torch.autograd.grad(
            u, T_flat, grad_outputs=torch.ones_like(u), create_graph=True
        )[0]
        u_x = torch.autograd.grad(
            u, X_flat, grad_outputs=torch.ones_like(u), create_graph=True
        )[0]
        u_xx = torch.autograd.grad(
            u_x, X_flat, grad_outputs=torch.ones_like(u_x), create_graph=True
        )[0]
        return torch.mean((u_t - u_xx - f(X_flat, T_flat)) ** 2)

    def initial_condition_loss(self, x_ic: torch.Tensor, u0: callable) -> torch.Tensor:
        """
        Compute the loss for the initial condition.
        Args:
            x_ic: Spatial input tensor for the initial condition.
            u0: Function representing the initial condition.
        Returns:
            Mean squared error loss for the initial condition.
        """
        u0_pred = self.model(
            x_ic.to(self.device), torch.zeros_like(x_ic).to(self.device)
        )
        return torch.mean((u0_pred - u0(x_ic.to(self.device))) ** 2)

    def boundary_condition_loss(
        self,
        t_bc: torch.Tensor,
        boundary_conditions: list,
        interval: tuple,
        mode: str = "uniform",
    ) -> torch.Tensor:
        """
        Compute the loss for the boundary conditions.
        Args:
            t_bc: Temporal input tensor for the boundary conditions.
            boundary_conditions: List of boundary condition values at the left and right boundaries.
            interval: Tuple representing the spatial interval [x0, xL].
            mode: String representing the mode of the boundary conditions.
        Returns:
            Mean squared error loss for the boundary conditions.
        """
        t_dev = t_bc.to(self.device)

        if mode == "sawtooth":
            x1 = x_min_sawtooth(t_dev)
            x2 = x_max_sawtooth(t_dev)
        elif mode == "senoidal":
            x1 = x_min_sin(t_dev)
            x2 = x_max_sin(t_dev)
        elif mode == "squarewave":
            x1 = x_min_sq(t_dev)
            x2 = x_max_sq(t_dev)
        else:
            x0, xL = interval
            x1 = torch.full_like(t_dev, x0)
            x2 = torch.full_like(t_dev, xL)

        u_bc1 = self.model(x1, t_dev)
        u_bc2 = self.model(x2, t_dev)
        return torch.mean((u_bc1 - boundary_conditions[0]) ** 2) + torch.mean(
            (u_bc2 - boundary_conditions[1]) ** 2
        )

    def loss(
        self,
        X_flat: torch.Tensor,
        T_flat: torch.Tensor,
        x_ic: torch.Tensor,
        t_bc: torch.Tensor,
        weight_factors: list,
        boundary_cond: list,
        u0: callable,
        f: callable,
        interval: tuple,
        mode: str,
    ) -> torch.Tensor:
        """
        Compute the total loss for the PINN.
        Args:
            X_flat: Flattened spatial input tensor.
            T_flat: Flattened temporal input tensor.
            x_ic: Spatial input tensor for the initial condition.
            t_bc: Temporal input tensor for the boundary conditions.
            weight_factors: List of weight factors for the losses.
            boundary_cond: List of boundary condition values at the left and right boundaries.
            u0: Function representing the initial condition.
            f: Function representing the source term in the PDE.
            interval: Tuple representing the spatial interval [x0, xL].
            mode: String representing the mode of the boundary conditions.
        Returns:
            Total loss for the PINN, which is a weighted sum of the PDE loss, initial
            condition loss, and boundary condition loss.
        """
        w1, w2, w3 = weight_factors
        Lpde = self.heat_equation_loss(X_flat, T_flat, f)
        Li = self.initial_condition_loss(x_ic, u0)
        Lb = self.boundary_condition_loss(t_bc, boundary_cond, interval, mode)
        return w1 * Lpde + w2 * Li + w3 * Lb

    def train(
        self,
        epochs: int,
        print_every: int,
        data: list,
        weight_factors: list,
        boundary_cond: list,
        u0: callable,
        f: callable,
        interval: tuple,
        mode: str,
    ):
        """
        Train the PINN model.

        Args:
            epochs: Number of training epochs.
            print_every: Frequency of printing the loss during training.
            data: List containing the training data [X_flat, T_flat, x_ic, t_bc].
            weight_factors: List of weight factors for the losses.
            boundary_cond: List of boundary condition values at the left and right boundaries.
            u0: Function representing the initial condition.
            f: Function representing the source term in the PDE.
            interval: Tuple representing the spatial interval [x0, xL].
            mode: String representing the mode of the boundary conditions.
        """
        X_flat, T_flat, x_ic, t_bc = data
        X_flat, T_flat = X_flat.to(self.device), T_flat.to(self.device)
        x_ic, t_bc = x_ic.to(self.device), t_bc.to(self.device)
        X_flat.requires_grad = True
        T_flat.requires_grad = True
        # No need to activate gradients for x_ic and t_bc

        for epoch in range(1, epochs + 1):
            self.optimizer.zero_grad()
            loss_val = self.loss(
                X_flat,
                T_flat,
                x_ic,
                t_bc,
                weight_factors,
                boundary_cond,
                u0,
                f,
                interval,
                mode,
            )
            loss_val.backward()
            self.optimizer.step()
            if epoch % print_every == 0 or epoch == epochs:
                print(f"Epoch {epoch}, Loss: {loss_val.item():.6f}")
            self.loss_values["heat_eq_loss"].append(
                self.heat_equation_loss(X_flat, T_flat, f).item() * weight_factors[0]
            )
            self.loss_values["initial_condition_loss"].append(
                self.initial_condition_loss(x_ic, u0).item() * weight_factors[1]
            )
            self.loss_values["boundary_condition_loss"].append(
                self.boundary_condition_loss(t_bc, boundary_cond, interval, mode).item()
                * weight_factors[2]
            )
            self.loss_values["total_loss"].append(loss_val.item())

    def predict(self, x: torch.Tensor, t: torch.Tensor) -> np.ndarray:
        """
        Predict the solution of the heat equation at given spatial and temporal points.
        Args:
            x: Spatial input tensor.
            t: Temporal input tensor.
        Returns:
            Numpy array of predicted values at the given points.
        """
        x, t = x.to(self.device), t.to(self.device)
        with torch.no_grad():
            return self.model(x, t).cpu().numpy()

    def draw_loss(self):
        """
        Draw the loss values during training.
        """
        plt.figure(figsize=(8, 5))
        plt.plot(self.loss_values["heat_eq_loss"], label="PDE Loss")
        plt.plot(self.loss_values["initial_condition_loss"], label="IC Loss")
        plt.plot(self.loss_values["boundary_condition_loss"], label="BC Loss")
        plt.yscale("log")
        plt.xlabel("Épocas")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid()
        plt.show()


def x_min_sawtooth(t: torch.Tensor) -> torch.Tensor:
    """
    Irregular left boundary: sawtooth function of t

    Args:
        t (torch.Tensor): Time tensor.
    Returns:
        torch.Tensor: Values of the left boundary at time t.
    """
    theta = 2 * torch.pi * t / lambda_x
    frac = (theta / (2 * torch.pi)) - torch.floor(theta / (2 * torch.pi))
    saw = 2 * (frac - 0.5)
    return interval[0] + A_x * saw


def x_max_sawtooth(t: torch.Tensor) -> torch.Tensor:
    """
    Irregular right boundary: sawtooth function of t

    Args:
        t (torch.Tensor): Time tensor.
    Returns:
        torch.Tensor: Values of the right boundary at time t.
    """
    theta = 2 * torch.pi * t / lambda_x
    frac = (theta / (2 * torch.pi)) - torch.floor(theta / (2 * torch.pi))
    saw = 2 * (frac - 0.5)
    return interval[1] + A_x * saw


def x_min_sin(t: torch.Tensor) -> torch.Tensor:
    """
    Irregular left boundary: sine function of t

    Args:
        t (torch.Tensor): Time tensor.
    Returns:
        torch.Tensor: Values of the left boundary at time t.
    """
    return interval[0] + A_x * torch.sin(2 * torch.pi * t / lambda_x)


def x_max_sin(t: torch.Tensor) -> torch.Tensor:
    """
    Irregular right boundary: sine function of t

    Args:
        t (torch.Tensor): Time tensor.
    Returns:
        torch.Tensor: Values of the right boundary at time t.
    """
    return interval[1] + A_x * torch.sin(2 * torch.pi * t / lambda_x)


def x_min_sq(t: torch.Tensor) -> torch.Tensor:
    """
    Irregular left boundary: square wave function of t

    Args:
        t (torch.Tensor): Time tensor.
    Returns:
        torch.Tensor: Values of the left boundary at time t.
    """
    # Calculate the fractional part of t with respect to the period λx
    theta = torch.fmod(t, lambda_x)  # tensor with values in [0, λx)
    frac = theta / lambda_x  # fractional part in [0, 1)
    # If frac < 0.5 → +1, if frac >= 0.5 → -1
    wave = torch.where(frac < 0.5, torch.ones_like(frac), -torch.ones_like(frac))
    return interval[0] + A_x * wave


def x_max_sq(t: torch.Tensor) -> torch.Tensor:
    """
    Irregular right boundary: square wave function of t

    Args:
        t (torch.Tensor): Time tensor.
    Returns:
        torch.Tensor: Values of the right boundary at time t.
    """
    theta = torch.fmod(t, lambda_x)
    frac = theta / lambda_x
    wave = torch.where(frac < 0.5, torch.ones_like(frac), -torch.ones_like(frac))
    return interval[1] + A_x * wave


def load_data(
    interval: list,
    mesh_points: int,
    time_steps: int,
    reduced_mesh_points: int,
    t_max: float,
    bc_points: int,
    ic_points: int,
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Loads the training data for the PINN.
    Args:
        interval (list): Spatial interval [x0, xL].
        mesh_points (int): Number of points in the spatial mesh.
        time_steps (int): Number of time steps.
        reduced_mesh_points (int): Reduced number of points for training.
        t_max (float): Maximum time value.
        bc_points (int): Multiplier for the number of boundary condition points.
        ic_points (int): Multiplier for the number of initial condition points.
        mode (str): Mode for generating the data ("uniform", "random", "sawtooth", "senoidal", "squarewave").
    Returns:
        X_train (torch.Tensor): Training data for spatial points.
        T_train (torch.Tensor): Training data for time points.
        x_ic (torch.Tensor): Initial condition points in space.
        t_bc (torch.Tensor): Boundary condition points in time.
    """

    if mode == "uniform":
        x_ic = torch.linspace(interval[0], interval[1], mesh_points).unsqueeze(1)
        t_bc = torch.linspace(0, t_max, time_steps).unsqueeze(1)
        X, T = torch.meshgrid(
            x_ic[:reduced_mesh_points].squeeze(), t_bc.squeeze(), indexing="ij"
        )
        X_train, T_train = X.flatten().unsqueeze(1), T.flatten().unsqueeze(1)
        x_ic = torch.linspace(
            interval[0], interval[1], mesh_points * ic_points
        ).unsqueeze(1)

    elif mode == "random":
        N = reduced_mesh_points * time_steps
        X_train = (interval[1] - interval[0]) * torch.rand(N, 1) + interval[0]
        T_train = t_max * torch.rand(N, 1)
        x_ic = torch.linspace(
            interval[0], interval[1], mesh_points * ic_points
        ).unsqueeze(1)

    elif mode == "sawtooth":
        N = reduced_mesh_points * time_steps
        X_list, T_list = [], []
        while len(X_list) < N:
            need = N - len(X_list)
            x_c = (
                interval[0]
                - 0.5
                + (interval[1] - interval[0] + 1) * torch.rand(need, 1)
            )
            t_c = t_max * torch.rand(need, 1)

            # We filter x_c to be within the sawtooth wave boundaries
            mask = (x_c >= x_min_sawtooth(t_c)) & (x_c <= x_max_sawtooth(t_c))
            X_list.extend(x_c[mask].tolist())
            T_list.extend(t_c[mask].tolist())

        X_train = torch.tensor(X_list[:N]).unsqueeze(1)
        T_train = torch.tensor(T_list[:N]).unsqueeze(1)

        # Initial condition at t=0, in [x_min(0), x_max(0)]
        x0 = x_min_sawtooth(torch.tensor(0.0)).item()
        x1 = x_max_sawtooth(torch.tensor(0.0)).item()
        x_ic = torch.linspace(x0, x1, mesh_points * ic_points).unsqueeze(1)

        # We attempt to generate t_bc with mesh_points points per jump, but when evaluating
        # in the loss function, the function has a value at each t_bc and no matter how
        # many times we repeat the times, it will not generate more points at the sawtooth
        # jumps; it will simply generate the same point many times. This is what the
        # following code does.

        # x_min_vals = x_min(t_bc)
        # dx = x_min_vals[1:] - x_min_vals[:-1]
        # # consider a jump if change > jump_tol
        # jump_tol = A_x  # half cycle
        # jump_mask = (dx.abs() > jump_tol).flatten()

        # # 5) Repeat each jump time mesh_points times
        # if jump_mask.any():
        #     t_jumps = t_bc[:-1][jump_mask]            # (H, 1)
        #     T_ext = t_jumps.repeat_interleave(mesh_points, dim=0)
        #     t_bc = torch.cat([t_bc, T_ext], dim=0)

    elif mode == "senoidal":
        N = reduced_mesh_points * time_steps
        X_list, T_list = [], []

        while len(X_list) < N:
            need = N - len(X_list)
            # We sample x_c in [interval[0] - A_x, interval[1] + A_x] because the sine wave
            # oscillates between these values.
            x_c = (interval[0] - A_x) + (
                interval[1] - interval[0] + 2 * A_x
            ) * torch.rand(need, 1)
            t_c = t_max * torch.rand(need, 1)

            # We filter x_c to be within the sine wave boundaries
            mask = (x_c >= x_min_sin(t_c)) & (x_c <= x_max_sin(t_c))
            X_list.extend(x_c[mask].tolist())
            T_list.extend(t_c[mask].tolist())

        X_train = torch.tensor(X_list[:N]).unsqueeze(1)
        T_train = torch.tensor(T_list[:N]).unsqueeze(1)

        # Initial condition at t=0, in [x_min_sin(0), x_max_sin(0)]
        x0 = x_min_sin(torch.tensor(0.0)).item()
        x1 = x_max_sin(torch.tensor(0.0)).item()
        x_ic = torch.linspace(x0, x1, mesh_points * ic_points).unsqueeze(1)

    elif mode == "squarewave":
        N = reduced_mesh_points * time_steps
        X_list, T_list = [], []

        while len(X_list) < N:
            need = N - len(X_list)
            # We sample x_c in [interval[0] - A_x, interval[1] + A_x] because the square wave
            # oscillates between these values.
            x_c = (interval[0] - A_x) + (
                interval[1] - interval[0] + 2 * A_x
            ) * torch.rand(need, 1)
            t_c = t_max * torch.rand(need, 1)

            # We filter x_c to be within the square wave boundaries
            mask = (x_c >= x_min_sq(t_c)) & (x_c <= x_max_sq(t_c))
            X_list.extend(x_c[mask].tolist())
            T_list.extend(t_c[mask].tolist())

        X_train = torch.tensor(X_list[:N]).unsqueeze(1)  # dx = 1
        T_train = torch.tensor(T_list[:N]).unsqueeze(1)

        # Initial condition at t=0, in [x_min_sq(0), x_max_sq(0)]
        x0 = x_min_sq(torch.tensor(0.0)).item()
        x1 = x_max_sq(torch.tensor(0.0)).item()
        x_ic = torch.linspace(x0, x1, mesh_points * ic_points).unsqueeze(1)

    else:
        raise ValueError(f"Modo desconocido: {mode}")

    t_bc = torch.linspace(0, t_max, time_steps * bc_points).unsqueeze(1)

    return X_train, T_train, x_ic, t_bc


def draw_solution_irregular(
    X_pred: np.ndarray,
    T_pred: np.ndarray,
    u_pred_uniform: np.ndarray,
    X_train: np.ndarray,
    T_train: np.ndarray,
    u_pred_train: np.ndarray,
    x_ic: np.ndarray,
    u_pred_ic: np.ndarray,
    t_bc: np.ndarray,
    x_min_bc: np.ndarray,
    u_pred_bc_min: np.ndarray,
    x_max_ic: np.ndarray,
    u_pred_bc_max: np.ndarray,
):
    """
    Plot the solution of the heat equation in an irregular domain.

    Args:
        X_pred: Points of prediction in x (always uniform)
        T_pred: Points of prediction in t (always uniform)
        u_pred_uniform: Predicted solution at the prediction points
        X_train: Training points in x (can be random)
        T_train: Training points in t (can be random)
        u_pred_train: Predicted solution at the training points
        x_ic: Points of initial condition in x
        u_pred_ic: Predicted solution at the initial condition points
        t_bc: Points of boundary condition in t
        x_min_bc: Left boundary points in x
        u_pred_bc_min: Predicted solution at the left boundary points
        x_max_ic: Right boundary points in x
        u_pred_bc_max: Predicted solution at the right boundary points
    """

    fig = go.Figure()

    # 1) Semi-transparent surface drawn with PINN PREDICT UNIFORM data
    fig = go.Figure(
        data=go.Scatter3d(
            x=X_pred,
            y=T_pred,
            z=u_pred_uniform.flatten(),
            mode="markers",
            marker=dict(size=1, color="blue", opacity=0.3),
            name="Predicted surface",
        )
    )

    # 2) Training points
    fig.add_trace(
        go.Scatter3d(
            x=X_train.flatten(),
            y=T_train.flatten(),
            z=u_pred_train.flatten(),
            mode="markers",
            marker=dict(size=1, color="red", opacity=0.8),
            name="Training",
        )
    )

    # 3) Initial condition (t=0)
    fig.add_trace(
        go.Scatter3d(
            x=x_ic.flatten(),
            y=[0] * len(x_ic),
            z=u_pred_ic.flatten(),
            mode="markers",
            marker=dict(size=3, color="blue", opacity=0.9),
            name="Initial condition",
        )
    )

    # 4a) Left boundary: x_min_bc
    fig.add_trace(
        go.Scatter3d(
            x=x_min_bc.flatten(),
            y=t_bc.flatten(),
            z=u_pred_bc_min.flatten(),
            mode="markers",
            marker=dict(size=3, color="green", opacity=0.8),
            name="Left boundary",
        )
    )

    # 4b) Right boundary: x_max_ic
    fig.add_trace(
        go.Scatter3d(
            x=x_max_ic.flatten(),
            y=t_bc.flatten(),
            z=u_pred_bc_max.flatten(),
            mode="markers",
            marker=dict(size=3, color="orange", opacity=0.8),
            name="Right boundary",
        )
    )

    fig.update_layout(
        title="PINN solution in irregular domain",
        scene=dict(xaxis_title="x", yaxis_title="t", zaxis_title="u(x,t)"),
        legend=dict(x=0.15, y=0.98, bgcolor="rgba(255,255,255,0.7)"),
    )

    fig.show()


def draw_solution_regular(
    X_pred: np.ndarray,
    T_pred: np.ndarray,
    u_pred_uniform: np.ndarray,
    X_train: np.ndarray,
    T_train: np.ndarray,
    u_pred_train: np.ndarray,
    x_ic: np.ndarray,
    t_bc: np.ndarray,
    mesh_points: int,
    time_steps: int,
    bc_points: int,
    ic_points: int,
):
    """
    Plot the solution of the heat equation in a regular domain.

    Args:
        X_pred: Points of prediction in x (uniformly spaced)
        T_pred: Points of prediction in t (uniformly spaced)
        u_pred: Predicted solution at the prediction points
        X_train: Training points in x (can be random)
        T_train: Training points in t (can be random)
        u_pred_train: Predicted solution at the training points
        x_ic: Points of initial condition in x
        t_bc: Points of boundary condition in t
        mesh_points: Number of mesh points in the spatial domain
        time_steps: Number of time steps in the temporal domain
        bc_points: Multiplier of time_steps to have more points at the boundary
        ic_points: Multiplier of mesh_points to have more points at the initial cond
    """

    # Flatten the arrays for the scatter plot
    u_pred_T = u_pred_uniform.T.flatten()
    u_pred_uniform = u_pred_uniform.flatten()
    u_pred_train = u_pred_train.flatten()

    x_ic = np.tile(x_ic, time_steps * bc_points)  # Repeat x_ic for each value of t
    t_bc = np.repeat(t_bc, mesh_points * ic_points)  # Repeat t_bc for each value of x

    fig_plotly = go.Figure()

    # 1) The semi‑transparent surface drawn with PINN PREDICT UNIFORM data
    fig_plotly.add_trace(
        go.Mesh3d(
            x=X_pred,
            y=T_pred,
            z=u_pred_uniform,
            colorscale="Blues",
            opacity=0.5,
            name="PINN",
            showlegend=True,
        )
    )

    # 2) Training points
    fig_plotly.add_trace(
        go.Scatter3d(
            x=X_train,
            y=T_train,
            z=u_pred_train,
            mode="markers",
            marker=dict(size=1, color="red", opacity=0.8),
            name="Training points",
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
            name="Initial condition (t=0)",
            showlegend=True,
        )
    )

    # 4) Sub‑sample along x_min and x_max
    slice_min = np.arange(0, x_ic.shape[0], mesh_points * ic_points)
    slice_max = np.arange(
        mesh_points * ic_points - 1, x_ic.shape[0], mesh_points * ic_points
    )
    both_slices = np.concatenate([slice_min, slice_max])

    # Draw them in one trace:
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
            x=0.2,
            y=0.98,
            bgcolor="rgba(255,255,255,0.7)",  # semi‑transparent white background
        ),
    )

    fig_plotly.update_traces(
        hovertemplate="x: %{x:.4f}<br>" "t: %{y:.4f}<br>" "u: %{z:.4f}<extra></extra>"
    )

    fig_plotly.show()


if __name__ == "__main__":
    # ---------------- Problem Definition ----------------
    boundary_conditions = [0, 0]  # Boundary conditions at x=0 and x=L
    f = lambda x, t: torch.sin(np.pi * x / 2) * torch.exp(-10 * t)  # Source term
    u0 = lambda x: torch.sin(np.pi * x / 2)  # Initial condition

    # ---------------- Spatial and Temporal Discretization ----------------
    interval = [0, 4]
    dt = 0.1
    t_max = 2
    time_steps = int(t_max / dt) + 1
    mesh_points = 41
    data_percentage = 0.6  # Percentage of data to use for training (0.2 = 20%)
    reduced_mesh_points = round(mesh_points * data_percentage)
    bc_points = 1  # Multiplier of time_steps to have more points at the boundary
    ic_points = 1  # Multiplier of mesh_points to have more points at the initial cond

    mode = "random"  # "uniform", "random", "sawtooth", "senoidal" or "squarewave"

    # ---------------- Parameters for irregular domain in x ------------
    A_x = 0.25  # Amplitude of the boundary in x
    lambda_x = 1  # Period in t

    X_train, T_train, x_ic, t_bc = load_data(
        interval,
        mesh_points,
        time_steps,
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
    print_every = 500
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

    if mode in ["sawtooth", "senoidal", "squarewave"]:
        # 1) Predict in the training points
        u_pred_train = pinn.predict(X_train, T_train)

        # 2) Predict in a uniform grid in the irregular domain
        num_x, num_t = int(50 * (interval[1] - interval[0])), int(100 * t_max)
        x_lin = np.linspace(interval[0] - A_x, interval[1] + A_x, num_x)
        t_lin = np.linspace(0, t_max, num_t)
        Xg, Tg = np.meshgrid(x_lin, t_lin)
        Xg_flat = Xg.reshape(-1, 1)
        Tg_flat = Tg.reshape(-1, 1)
        Tg_tensor = torch.from_numpy(Tg_flat).float().to(pinn.device)
        if mode == "sawtooth":
            x_min_vals = x_min_sawtooth(Tg_tensor).cpu().numpy()
            x_max_vals = x_max_sawtooth(Tg_tensor).cpu().numpy()
        elif mode == "senoidal":
            x_min_vals = x_min_sin(Tg_tensor).cpu().numpy()
            x_max_vals = x_max_sin(Tg_tensor).cpu().numpy()
        elif mode == "squarewave":
            x_min_vals = x_min_sq(Tg_tensor).cpu().numpy()
            x_max_vals = x_max_sq(Tg_tensor).cpu().numpy()
        mask = (Xg_flat >= x_min_vals) & (Xg_flat <= x_max_vals)
        X_pred = torch.from_numpy(Xg_flat[mask]).float().to(pinn.device)
        T_pred = torch.from_numpy(Tg_flat[mask]).float().to(pinn.device)
        X_pred = X_pred.flatten().unsqueeze(1)
        T_pred = T_pred.flatten().unsqueeze(1)
        u_pred_uniform = pinn.predict(X_pred, T_pred)

        # 3) Predict for initial condition at t=0
        t_zero = torch.zeros_like(x_ic).to(pinn.device)
        u_pred_ic = pinn.predict(x_ic, t_zero)

        # 4) Predict for irregular boundary conditions
        t_bc_tensor = t_bc.to(pinn.device)
        if mode == "sawtooth":
            x_min_bc = x_min_sawtooth(t_bc_tensor)
            x_max_ic = x_max_sawtooth(t_bc_tensor)
        elif mode == "senoidal":
            x_min_bc = x_min_sin(t_bc_tensor)
            x_max_ic = x_max_sin(t_bc_tensor)
        elif mode == "squarewave":
            x_min_bc = x_min_sq(t_bc_tensor)
            x_max_ic = x_max_sq(t_bc_tensor)
        u_pred_bc_min = pinn.predict(x_min_bc, t_bc_tensor)
        u_pred_bc_max = pinn.predict(x_max_ic, t_bc_tensor)

        x_ic = x_ic.detach().numpy().flatten()
        t_bc = t_bc.detach().numpy().flatten()
        X_train = X_train.detach().numpy().flatten()
        T_train = T_train.detach().numpy().flatten()
        X_pred = X_pred.detach().numpy().flatten()
        T_pred = T_pred.detach().numpy().flatten()
        x_min_bc = x_min_bc.detach().numpy().flatten()
        x_max_ic = x_max_ic.detach().numpy().flatten()

        # Plot the solution
        draw_solution_irregular(
            X_pred,
            T_pred,
            u_pred_uniform,
            X_train,
            T_train,
            u_pred_train,
            x_ic,
            u_pred_ic,
            t_bc,
            x_min_bc,
            u_pred_bc_min,
            x_max_ic,
            u_pred_bc_max,
        )

    elif mode in ["uniform", "random"]:
        # Define where we want the network to PREDICT
        X, T = torch.meshgrid(x_ic.squeeze(), t_bc.squeeze(), indexing="ij")
        X_pred, T_pred = X.flatten().unsqueeze(1), T.flatten().unsqueeze(1)

        # To plot correctly I need a prediction that allows me to paint the surface or
        # mesh and on the other hand I need another prediction to know the value of my
        # random training points. If I just want to know the value of the prediction
        # points or if they coincide with the training ones, a predict is enough. But if
        # not, I would need to make another predict with the uniformly spaced points.

        # Prediction on one hand with the points we have trained and on the other hand
        # with uniformly spaced points to be able to paint the function well

        # Prediction of the original training points
        u_pred_train = pinn.predict(X_train, T_train).reshape(
            reduced_mesh_points, time_steps
        )

        # Prediction of the new uniformly distributed points
        u_pred = pinn.predict(X_pred, T_pred).reshape(
            mesh_points * ic_points, time_steps * bc_points
        )

        # ---------------- Plot ----------------
        x_ic = x_ic.detach().numpy().flatten()
        t_bc = t_bc.detach().numpy().flatten()
        X_train = X_train.detach().numpy().flatten()
        T_train = T_train.detach().numpy().flatten()
        X_pred = X_pred.detach().numpy().flatten()
        T_pred = T_pred.detach().numpy().flatten()

        f = lambda x, t: np.sin(np.pi * x / 2) * np.exp(-10 * t)  # Source term in numpy
        MSE_offgrid_splines_pinn = residual_MSE_two_stage(
            x_ic, t_bc, u_pred.T, f, num_samples=400
        )
        print(
            f"Off-grid MSE with cubic splines with PINN: {MSE_offgrid_splines_pinn:.8f}"
        )

        draw_solution_regular(
            X_pred,
            T_pred,
            u_pred,
            X_train,
            T_train,
            u_pred_train,
            x_ic,
            t_bc,
            mesh_points,
            time_steps,
            bc_points,
            ic_points,
        )

    pinn.draw_loss()
