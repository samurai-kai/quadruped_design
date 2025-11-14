# Kai De La Cruz
# 3DOF Leg Static Load Analysis using Jacobian Method (Imperial Units)
# Date: 10/07/2025

import numpy as np
import sympy as sp

# ---------- DH Transform ----------
def dh_transform_sym(theta, d, a, alpha):
    """Standard Denavit–Hartenberg transformation (symbolic)."""
    ct, st = sp.cos(theta), sp.sin(theta)
    ca, sa = sp.cos(alpha), sp.sin(alpha)
    T = sp.Matrix([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0,      sa,     ca,   d],
        [0,       0,      0,   1]
    ])
    return sp.simplify(T)

# ---------- Frame Composition ----------
def frame_tf(Tn, Tm):
    """Compose two DH transforms: Tnm = Tm @ Tn."""
    return Tm @ Tn

# ---------- Extract z-axis and origin ----------
def output_z_p(T):
    """Extract z-axis (rotation axis) and origin position vector from T."""
    z = T[0:3, 2]
    p = T[0:3, 3]
    return z, p

# ---------- Jacobian ----------
def jacobian_sym(z_list, p_list, p_e):
    """Build full 6×n geometric Jacobian."""
    Jp = sp.Matrix.hstack(*[z_list[i].cross(p_e - p_list[i]) for i in range(len(z_list))])
    Jo = sp.Matrix.hstack(*[z_list[i] for i in range(len(z_list))])
    return sp.Matrix.vstack(Jp, Jo)

# ---------- Torque computation ----------
def torque_from_force(J, F):
    """Joint torques via Jacobian transpose method."""
    return J.T @ F


# ======================================================
# ====================== MAIN ==========================
# ======================================================
if __name__ == "__main__":
    # --- Symbolic variables ---
    theta1, theta2, theta3 = sp.symbols('theta1 theta2 theta3')
    L1, L2, L3, L4 = sp.symbols('L1 L2 L3 L4')

    # --- DH setup (matches your NE model) ---
    T01 = dh_transform_sym(np.pi/2, 0, 0, np.pi/2)
    T12 = dh_transform_sym(theta1 - sp.pi/2, L1, 0, -sp.pi/2)
    T23 = dh_transform_sym(theta2 - sp.pi/2, L2, L3, 0)
    T3e = dh_transform_sym(theta3, 0, L4, 0)

    # --- Chain transformations ---
    T02 = frame_tf(T01, T12)
    T03 = frame_tf(T02, T23)
    T0e = frame_tf(T03, T3e)

    # --- Extract z and p for each joint ---
    z1, p1 = output_z_p(T01)
    z2, p2 = output_z_p(T02)
    z3, p3 = output_z_p(T03)
    _,  pe = output_z_p(T0e)

    # --- Build Jacobian (6×3) ---
    J = sp.simplify(jacobian_sym([z1, z2, z3], [p1, p2, p3], pe))

    # --- External wrench (force only) ---
    # 58 N = 13.03 lbf upward; we keep N for math, convert later
    F = sp.Matrix([0, 0, 200, 0, 0, 0])

    # --- Joint torques (Nm) ---
    tau = sp.simplify(torque_from_force(J, F))

    # --- Substitute numeric parameters ---
    subs = {
        theta1: np.deg2rad(0),
        theta2: np.deg2rad(45),
        theta3: np.deg2rad(90),
        L1: 0.061,
        L2: 0.083,
        L3: 0.146,
        L4: 0.165
    }
    tau_num = tau.evalf(subs=subs)

    N_to_lbf   = 0.224809        # 1 N = 0.224809 lbf
    Nm_to_ftlb = 0.73756214927727  # 1 N·m = 0.737562 ft·lbf

    # Convert
    tau_ftlb = tau_num * Nm_to_ftlb
    F_lbf = F * N_to_lbf

    print("\n" + "="*55)
    print("3DOF LEG STATIC LOAD ANALYSIS (JACOBIAN METHOD)")
    print("Results in Imperial Units")
    print("="*55)

    theta_vals_deg = [
        float(subs[theta1] * 180 / np.pi),
        float(subs[theta2] * 180 / np.pi),
        float(subs[theta3] * 180 / np.pi)
    ]

    print("\nJoint Angles (degrees):")
    print(f"θ1 = {theta_vals_deg[0]:.2f}°,  θ2 = {theta_vals_deg[1]:.2f}°,  θ3 = {theta_vals_deg[2]:.2f}°")


    print("\nApplied Foot Force [lbf]:")
    sp.pprint(F_lbf[0:3, :])  # only linear part

    print("\nJoint Torques [ft·lbf]:")
    sp.pprint(tau_ftlb)

    print("\nNotes:")
    print(" - Positive torque = actuator torque resisting external load.")
    print(" - Computed using Jacobian Transpose τ = Jᵀ·F.")
    print(" - Geometry & DH parameters identical to NE statics model.")
    print("="*55)
