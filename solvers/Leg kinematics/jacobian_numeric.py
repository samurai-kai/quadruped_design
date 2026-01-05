#!/usr/bin/env python3
"""
Numeric Jacobian for the 3-DOF quadruped leg.
Derived directly from the symbolic DH model provided by Kai.
"""

import numpy as np
import sympy as sp

# ===============================================================
# 1. Symbolic Variables
# ===============================================================

theta1, theta2, theta3 = sp.symbols('theta1 theta2 theta3')
L1, L2, L3, L4 = sp.symbols('L1 L2 L3 L4')

# ===============================================================
# 2. DH Transform (Symbolic)
# ===============================================================

def dh_transform_sym(theta, d, a, alpha):
    ct, st = sp.cos(theta), sp.sin(theta)
    ca, sa = sp.cos(alpha), sp.sin(alpha)
    return sp.Matrix([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [ 0,     sa,     ca,   d],
        [ 0,      0,      0,   1]
    ])

def frame_tf(Tn, Tm):
    """Compose transformations: Tnm = Tm @ Tn"""
    return Tm @ Tn

def output_z_p(T):
    """Extract joint rotation axis (z) and position (p) from a transform."""
    return T[0:3, 2], T[0:3, 3]

# ===============================================================
# 3. Build Symbolic Forward Kinematics
# ===============================================================

# Using your exact DH setup:
T01 = dh_transform_sym(np.pi/2,           0, 0, np.pi/2)
T12 = dh_transform_sym(theta1 - sp.pi/2, L1, 0, -sp.pi/2)
T23 = dh_transform_sym(theta2 - sp.pi/2, L2, L3, 0)
T3e = dh_transform_sym(theta3,           0, L4, 0)

# Compose transforms
T02 = frame_tf(T01, T12)
T03 = frame_tf(T02, T23)
T0e = frame_tf(T03, T3e)

# Extract joint axes & positions
z1, p1 = output_z_p(T01)
z2, p2 = output_z_p(T02)
z3, p3 = output_z_p(T03)
_,  pe = output_z_p(T0e)

# ===============================================================
# 4. Symbolic Jacobian Construction
# ===============================================================

def jacobian_sym(z_list, p_list, p_e):
    Jp = sp.Matrix.hstack(*[
        z_list[i].cross(p_e - p_list[i]) for i in range(len(z_list))
    ])
    Jo = sp.Matrix.hstack(*[
        z_list[i] for i in range(len(z_list))
    ])
    return sp.Matrix.vstack(Jp, Jo)

# Full 6×3 symbolic Jacobian
J_sym = jacobian_sym([z1, z2, z3], [p1, p2, p3], pe)

# ===============================================================
# 5. Lambdify: Convert symbolic Jacobian → FAST NumPy function
# ===============================================================

J_func = sp.lambdify(
    (theta1, theta2, theta3, L1, L2, L3, L4),
    J_sym,
    "numpy"
)

# ===============================================================
# 6. Robot Link Lengths (meters)
# ===============================================================

# These match your statics file exactly:
L1_VAL = 0.061
L2_VAL = 0.083
L3_VAL = 0.146
L4_VAL = 0.165

# ===============================================================
# 7. Public API: numeric Jacobian
# ===============================================================

def jacobian_numeric(q):
    """
    Returns the 6×3 Jacobian matrix as a NumPy array for joint configuration q = [hip, thigh, knee].
    q values must be in radians.
    """
    q1, q2, q3 = q

    J = J_func(
        q1, q2, q3,
        L1_VAL, L2_VAL, L3_VAL, L4_VAL
    )

    # Ensure a clean NumPy float array
    return np.array(J, dtype=float)


# ===============================================================
# 8. Optional: Quick self-test
# ===============================================================
if __name__ == "__main__":
    # Test configuration (same as statics file)
    q_test = [0.0, np.deg2rad(45), np.deg2rad(90)]

    print("\nNumeric Jacobian at q =", q_test)
    print(jacobian_numeric(q_test))
