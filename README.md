This Bachelor´s Thesis repository implements and compares classical PDE solvers (finite differences and finite elements) with a Physics-Informed Neural Network (PINN) for the 1D heat equation.

## Requirements

Install the Python dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```
## Files

* **`traditional.py`**
  Implements FD and FEM solvers for Poisson, Laplace, and the heat equation.

  * Functions:

    * `poisson_fd`, `poisson_fem`: solve –u″=f
    * `laplace_fd`, `laplace_fem`: solve u″=0
    * `heat_equation_fd`: explicit FD in time + centered FD in space
    * `heat_equation_fem`: FEM in space + implicit Euler in time
    * Utilities for on-grid and off-grid MSE, bilinear and spline interpolation, and Matplotlib/Plotly plotting.
  * Run as a script to test any solver (edit parameters in the `__main__` block).

* **`pinn.py`**
  Defines a feed‐forward PINN architecture (`PINN`), plus `HeatEquationPINN` wrapper:

  * Methods for PDE loss, initial/boundary losses, training (`train`) and prediction (`predict`).
  * Logging of loss components and Plotly/Matplotlib helpers to visualize training and solution surfaces.
  * Run as a script to train and evaluate the PINN on the heat equation (edit parameters in the `__main__` block).

* **`combination.py`**
  Integrates `traditional.py` and `pinn.py` to:

  * Train the PINN and solve with FD/FEM under the same settings.
  * Measure run times and off‐grid errors.
  * Plot 3D comparisons of the “ground‐truth” classical solution and PINN predictions.
  * Run as a script to compare both approaches (edit parameters in the `__main__` block)

* **`requirements.txt`**
  Specifies package versions:

  ```
  matplotlib==3.9.2
  numpy==2.0.2
  plotly==6.0.0
  scipy==1.14.1
  torch==2.6.0
  ```

## Usage

* Run any of the included python files to obtain plots, visualisations, error metrics and time references for any of the specified PDE solving methods.
* Feel free to adjust mesh size, time step, PINN architecture, sampling mode or any other hyperparameter directly in each script.

For further clarifications and precise explanatory details, consult the pdf file containing the full BT project **`TFG_Ara_Adanez_Miguel.pdf`**
