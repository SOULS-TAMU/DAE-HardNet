import numpy as np
import pandas as pd

# --- Config ---
N_X1 = 50
N_X2 = 50
X1_MIN, X1_MAX = -1.0,  1.0
X2_MIN, X2_MAX = -1.0,  1.0
CSV_PATH = "dataset_with_grads.csv"

# --- Grid ---
x1 = np.linspace(X1_MIN, X1_MAX, N_X1)
x2 = np.linspace(X2_MIN, X2_MAX, N_X2)
X1, X2 = np.meshgrid(x1, x2, indexing="xy")

# --- Functions ---
Y1 = 6.0 * np.exp(2.0 * X1 + X2) + X1 * (X2 ** 3)

# First derivatives
dY1dx1 = 12.0 * np.exp(2.0 * X1 + X2) + (X2 ** 3)
dY1dx2 = 6.0 * np.exp(2.0 * X1 + X2) + 3.0 * X1 * (X2 ** 2)

# Second derivatives
d2Y1dx1 = 24.0 * np.exp(2.0 * X1 + X2)
d2Y1dx2 = 6.0 * np.exp(2.0 * X1 + X2) + 6.0 * X1 * X2
d2Y1dx1dx2 = 12.0 * np.exp(2.0 * X1 + X2) + 3.0 * (X2 ** 2)

# --- Flatten & Save ---
df = pd.DataFrame({
    "x1": X1.ravel(),
    "x2": X2.ravel(),
    "y1": Y1.ravel(),
    "d1y1dx1": dY1dx1.ravel(),
    "d1y1dx2": dY1dx2.ravel(),
    "d2y1dx1": d2Y1dx1.ravel(),
    "d2y1dx2": d2Y1dx2.ravel(),
    "d2y1dx1dx2": d2Y1dx1dx2.ravel()
})

assert len(df) == N_X1 * N_X2, "Unexpected number of rows"

df.to_csv(CSV_PATH, index=False)
print(f"Saved {len(df)} rows to {CSV_PATH}")
