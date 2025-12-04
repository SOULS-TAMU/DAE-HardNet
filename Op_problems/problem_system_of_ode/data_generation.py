# y12_dataset.py
# Generate y1, y2 and derivatives over x ∈ [-4, 4] with 1500 points and save to CSV.

import numpy as np
import pandas as pd

# ---- Constants (edit these if you want different values) ----
A = -2.0
B = -2.0
C = 0.5
E = 0.0

# ---- Grid ----
N = 1500
x = np.linspace(-4.0, 4.0, N)
rt3 = np.sqrt(3.0)

# ---- y1, y2 ----
# y1 = (1/18)x^4 - (5/9)x^2 - 8/27 + A sin(√3 x) + B cos(√3 x) + 2C + 2E x
y1 = (1.0/18.0)*x**4 - (5.0/9.0)*x**2 - (8.0/27.0) + A*np.sin(rt3*x) + B*np.cos(rt3*x) + 2.0*C + 2.0*E*x

# y2 = -(11/18)x^2 + (1/36)x^4 + 11/27 + 2A sin(√3 x) + 2B cos(√3 x) + C + E x
y2 = -(11.0/18.0)*x**2 + (1.0/36.0)*x**4 + (11.0/27.0) + 2.0*A*np.sin(rt3*x) + 2.0*B*np.cos(rt3*x) + C + E*x

# ---- First derivatives dy/dx ----
# dy1/dx = (2/9)x^3 - (10/9)x + √3 A cos(√3 x) - √3 B sin(√3 x) + 2E
dy1dx = (2.0/9.0)*x**3 - (10.0/9.0)*x + rt3*A*np.cos(rt3*x) - rt3*B*np.sin(rt3*x) + 2.0*E

# dy2/dx = (1/9)x^3 - (11/9)x + 2√3 A cos(√3 x) - 2√3 B sin(√3 x) + E
dy2dx = (1.0/9.0)*x**3 - (11.0/9.0)*x + 2.0*rt3*A*np.cos(rt3*x) - 2.0*rt3*B*np.sin(rt3*x) + E

# ---- Second derivatives d2y/dx ----
# d2y1/dx2 = (2/3)x^2 - (10/9) - 3A sin(√3 x) - 3B cos(√3 x)
d2y1dx = (2.0/3.0)*x**2 - (10.0/9.0) - 3.0*A*np.sin(rt3*x) - 3.0*B*np.cos(rt3*x)

# d2y2/dx2 = (1/3)x^2 - (11/9) - 6A sin(√3 x) - 6B cos(√3 x)
d2y2dx = (1.0/3.0)*x**2 - (11.0/9.0) - 6.0*A*np.sin(rt3*x) - 6.0*B*np.cos(rt3*x)

# ---- Save to CSV ----
df = pd.DataFrame({
    "x1": x,
    "y1": y1,
    "y2": y2,
    "dy1dx1": dy1dx,
    "dy2dx1": dy2dx,
    "d2y1dx1": d2y1dx,   # second derivative of y1 w.r.t x
    "d2y2dx1": d2y2dx    # second derivative of y2 w.r.t x
})

out_path = "dataset_with_grads.csv"
df.to_csv(out_path, index=False)
print(f"Dataset saved to {out_path}")
