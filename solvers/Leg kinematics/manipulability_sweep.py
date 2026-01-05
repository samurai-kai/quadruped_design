#!/usr/bin/env python3
import numpy as np
import math

# Import your numeric jacobian function
from jacobian_numeric import jacobian_numeric


# ===============================================================
#  JOINT LIMITS (DEGREES) — these should match your safety file
# ===============================================================

HIP_MIN_DEG,   HIP_MAX_DEG   = -85, 85
THIGH_MIN_DEG, THIGH_MAX_DEG = 0, 360
KNEE_MIN_DEG,  KNEE_MAX_DEG  = 0, 135   # knee never fully straight

# Convert once to radians
HIP_MIN,   HIP_MAX   = np.radians([HIP_MIN_DEG,   HIP_MAX_DEG])
THIGH_MIN, THIGH_MAX = np.radians([THIGH_MIN_DEG, THIGH_MAX_DEG])
KNEE_MIN,  KNEE_MAX  = np.radians([KNEE_MIN_DEG,  KNEE_MAX_DEG])

# ===============================================================
#  Yoshikawa Manipulability
# ===============================================================
def manipulability(J):
    """Compute Yoshikawa manipulability measure."""
    Jv = J[0:3, :]       # use linear velocity rows only
    M = Jv @ Jv.T
    det_val = np.linalg.det(M)
    return math.sqrt(max(det_val, 0.0))


# ===============================================================
#  Sweep Resolution (degrees)
# ===============================================================
HIP_STEP   = 5
THIGH_STEP = 5
KNEE_STEP  = 5


# ===============================================================
#  Sweeping joint space for singularities
# ===============================================================
def sweep_manipulability():
    singularities = []   # store configurations where w ≈ 0
    near_singular = []   # store low manipulability configs

    MANIP_THRESHOLD = 0.002  # low manipulability threshold

    print("\nSweeping joint limits for singularities...\n")

    for h_deg in range(HIP_MIN_DEG, HIP_MAX_DEG+1, HIP_STEP):
        for t_deg in range(THIGH_MIN_DEG, THIGH_MAX_DEG+1, THIGH_STEP):
            for k_deg in range(KNEE_MIN_DEG, KNEE_MAX_DEG+1, KNEE_STEP):

                # Convert to radians
                h = math.radians(h_deg)
                t = math.radians(t_deg)
                k = math.radians(k_deg)
                q = [h, t, k]

                # Compute Jacobian + manipulability
                J = jacobian_numeric(q)
                w = manipulability(J)

                if w == 0:
                    singularities.append((h_deg, t_deg, k_deg))
                elif w < MANIP_THRESHOLD:
                    near_singular.append((h_deg, t_deg, k_deg, w))

    return singularities, near_singular


# ===============================================================
#  Main
# ===============================================================
if __name__ == "__main__":
    singularities, near_singular = sweep_manipulability()

    print("\n===============================")
    print(" EXACT SINGULARITIES FOUND")
    print("===============================")
    if len(singularities) == 0:
        print("No exact singularities detected (w = 0)")
    else:
        for s in singularities:
            print(f"  q = {s} deg")

    print("\n===============================")
    print(" NEAR-SINGULAR CONFIGURATIONS")
    print("===============================")
    if len(near_singular) == 0:
        print("No near-singular configurations (w < threshold)")
    else:
        for s in near_singular:
            h, t, k, w = s
            print(f"  q = ({h}, {t}, {k}) deg → w = {w:.6f}")

    print("\nSweep complete.\n")
