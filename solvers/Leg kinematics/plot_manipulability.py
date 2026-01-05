#!/usr/bin/env python3
"""
Plot Yoshikawa manipulability with SAFE vs UNSAFE regions based on threshold.
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from jacobian_numeric import jacobian_numeric

# ===============================================================
# JOINT LIMITS (in degrees)
# ===============================================================

HIP_FIXED_DEG = 0

THIGH_MIN_DEG, THIGH_MAX_DEG = 0, 359
KNEE_MIN_DEG,  KNEE_MAX_DEG  = 0, 135

SAFE_THRESHOLD = 0.0008      # <-- YOU SET WHAT YOU CONSIDER SAFE

def deg(x): return math.radians(x)


# ===============================================================
# Yoshikawa manipulability
# ===============================================================
def manipulability(J):
    Jv = J[0:3, :]
    M = Jv @ Jv.T
    det_val = np.linalg.det(M)
    return math.sqrt(max(det_val, 0.0))


# ===============================================================
# Compute manipulability grid
# ===============================================================
def compute_manipulability_map():
    thigh_vals = np.arange(THIGH_MIN_DEG, THIGH_MAX_DEG + 1, 2)
    knee_vals  = np.arange(KNEE_MIN_DEG, KNEE_MAX_DEG + 1, 2)

    W = np.zeros((len(thigh_vals), len(knee_vals)))
    SAFE = np.zeros_like(W, dtype=bool)

    hip_rad = deg(HIP_FIXED_DEG)

    for i, t_deg in enumerate(thigh_vals):
        for j, k_deg in enumerate(knee_vals):

            q = [hip_rad, deg(t_deg), deg(k_deg)]
            J = jacobian_numeric(q)
            w = manipulability(J)

            W[i, j] = w
            SAFE[i, j] = w >= SAFE_THRESHOLD

    return thigh_vals, knee_vals, W, SAFE


# ===============================================================
# Plot heatmap with safe/unsafe threshold boundary
# ===============================================================
def plot_manipulability():
    thigh_vals, knee_vals, W, SAFE = compute_manipulability_map()

    plt.figure(figsize=(10, 6))

    # Main manipulability heatmap
    plt.imshow(
        W,
        extent=[knee_vals[0], knee_vals[-1], thigh_vals[0], thigh_vals[-1]],
        origin='lower',
        cmap='inferno',
        aspect='auto'
    )

    plt.colorbar(label="Manipulability")

    # ============================
    # Draw SAFE/UNSAFE contour line
    # ============================
    contour_levels = [SAFE_THRESHOLD]
    plt.contour(
        knee_vals,
        thigh_vals,
        W,
        levels=contour_levels,
        colors='cyan',
        linewidths=2,
        linestyles='dashed'
    )

    # # Mask unsafe regions (optional)
    # unsafe = np.ma.masked_where(SAFE, W)
    # plt.imshow(
    #     unsafe,
    #     extent=[knee_vals[0], knee_vals[-1], thigh_vals[0], thigh_vals[-1]],
    #     origin='lower',
    #     cmap='gray',
    #     alpha=0.5,
    #     aspect='auto'
    # )

    plt.title(
        f"Manipulability Heatmap with Safety Threshold\n"
        f"Hip fixed at {HIP_FIXED_DEG}°, Threshold = {SAFE_THRESHOLD}"
    )
    plt.xlabel("Knee angle (deg)")
    plt.ylabel("Thigh angle (deg)")

    plt.tight_layout()
    plt.show()


# ===============================================================
# MAIN
# ===============================================================
if __name__ == "__main__":
    print("\nPlotting manipulability with safety threshold...")
    plot_manipulability()
