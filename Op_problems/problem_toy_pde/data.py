########################################## Toy PDE: Heat Equation Dataset with Gradients ##########################################
import numpy as np
import pandas as pd

# Parameters
L = 5.0  # Length of the rod (x range)
alpha = 1.0  # Diffusion coefficient
t_min, t_max, t_points = 0.0, 10.0, 100
x_min, x_max, x_points = 0.0, L, 100
n = 5

# Precompute k
k = n * np.pi / L

# Analytical solution
def T_xt(x, t):
    return np.sin(k * x) * np.exp(-alpha * k**2 * t)

# Gradients
def dT_dt(x, t):
    return -alpha * k**2 * T_xt(x, t)

def dT_dx(x, t):
    return k * np.cos(k * x) * np.exp(-alpha * k**2 * t)

def d2T_dx2(x, t):
    return -k**2 * T_xt(x, t)

# Sampling grid
t_vals = np.linspace(t_min, t_max, t_points)
x_vals = np.linspace(x_min, x_max, x_points)

# Create dataset
rows = []
for t in t_vals:
    for x in x_vals:
        T_val = T_xt(x, t)
        dt_val = dT_dt(x, t)
        dx_val = dT_dx(x, t)
        dxx_val = d2T_dx2(x, t)
        # rows.append([t, x, T_val, dt_val, dx_val, dxx_val])
        rows.append([t, x, T_val])

# df = pd.DataFrame(rows, columns=["t", "x1", "y1", "dt", "dx", "dxx"])
df = pd.DataFrame(rows, columns=["t", "x1", "y1"])

# Save to CSV
# csv_path = "heat_equation_dataset_with_grads.csv"
csv_path = "dataset.csv"
df.to_csv(csv_path, index=False)
print(f"Saved dataset with gradients to {csv_path}")