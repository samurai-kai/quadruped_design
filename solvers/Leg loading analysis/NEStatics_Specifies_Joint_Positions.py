# Kai De La Cruz
# 3DOF Leg Static Load Analysis using Newton–Euler Method
# Evaluate forces, moments, and torques at a specific joint configuration

import sympy as sp
import numpy as np

# ---------- DH Transform ----------
def dh_transform_sym(theta, d, a, alpha):
    ct, st = sp.cos(theta), sp.sin(theta)
    ca, sa = sp.cos(alpha), sp.sin(alpha)
    return sp.Matrix([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0,      sa,     ca,   d],
        [0,       0,      0,   1]
    ])

# ---------- Frame Transformation ----------
def frame_tf(Tn, Tm):
    return Tm @ Tn

# ---------- Extract z & p ----------
def output_z_p(T):
    return T[0:3, 2], T[0:3, 3]

# ---------- Newton–Euler Statics ----------
def NE_statics(T_mats, masses, g_vec, F_ext):
    """Compute torques, reaction forces, and moments at each joint (base frame)."""
    # Forward kinematics
    T02 = frame_tf(T_mats[0], T_mats[1])
    T03 = frame_tf(T02, T_mats[2])
    T0e = frame_tf(T03, T_mats[3])

    z1, p1 = output_z_p(T_mats[0])
    z2, p2 = output_z_p(T02)
    z3, p3 = output_z_p(T03)
    _, pe = output_z_p(T0e)
    z, p = [z1, z2, z3], [p1, p2, p3]

    # COMs (midpoints approximation)
    pC1, pC2, pC3 = p1/2, (p1+p2)/2, (p2+pe)/2

    F_next = -F_ext
    N_next = sp.Matrix([0, 0, 0])
    tau, joint_forces, joint_moments = [], [], []

    for i in reversed(range(3)):
        mi = masses[i]
        F_i = F_next + mi * g_vec
        if i == 2:
            r_next = pe - p3
        elif i == 1:
            r_next = p3 - p2
        else:
            r_next = p2 - p1
        N_i = N_next + r_next.cross(F_next) + (pC1 if i==0 else pC2 if i==1 else pC3).cross(mi*g_vec)
        tau_i = z[i].dot(N_i)
        tau.insert(0, tau_i)
        joint_forces.insert(0, F_i)
        joint_moments.insert(0, N_i)
        F_next, N_next = F_i, N_i

    tau_vec = -sp.Matrix(tau)
    return tau_vec, joint_forces, joint_moments

# ---------- MAIN ----------
if __name__ == "__main__":
    # Symbols
    theta1, theta2, theta3 = sp.symbols('theta1 theta2 theta3')
    L1, L2, L3, L4 = sp.symbols('L1 L2 L3 L4')
    m1, m2, m3 = sp.symbols('m1 m2 m3')
    g = sp.Symbol('g')
    g_vec = sp.Matrix([0, 0, -g])

    # DH chain
    T01 = dh_transform_sym(sp.pi/2, 0, 0, sp.pi/2)
    T12 = dh_transform_sym(theta1 - sp.pi/2, L1, 0, -sp.pi/2)
    T23 = dh_transform_sym(theta2 - sp.pi/2, L2, L3, 0)
    T3e = dh_transform_sym(theta3, 0, L4, 0)

    # Constants
    F_ext = sp.Matrix([0, 0, 200])  # external upward force at foot
    masses = [0.5, 0.4, 0.3]
    subs_base = {L1: 0.061, L2: 0.083, L3: 0.146, L4: 0.165, g: 9.81, m1: 0.5, m2: 0.4, m3: 0.3}

    # ---------- Specify your joint angles (degrees) ----------
    # Example: θ₁ = 0°, θ₂ = 45°, θ₃ = 90°
    joint_angles_deg = [0, 45, 90]

    subs = subs_base.copy()
    subs.update({
        theta1: np.deg2rad(joint_angles_deg[0]),
        theta2: np.deg2rad(joint_angles_deg[1]),
        theta3: np.deg2rad(joint_angles_deg[2]),
    })

    # ---------- Compute results ----------
    tau_sym, F_list, N_list = NE_statics(
        [T01, T12, T23, T3e],
        masses,
        g_vec,
        F_ext
    )

    # Evaluate numerically
    tau_vals = tau_sym.evalf(subs=subs)
    F_joint_vals = [F.evalf(subs=subs) for F in F_list]
    N_joint_vals = [N.evalf(subs=subs) for N in N_list]

    # ---------- Convert to Imperial Units ----------
    N_to_lbf   = 0.224809
    Nm_to_ftlb = 0.73756214927727

    tau_ftlb = tau_vals * Nm_to_ftlb
    F_lbf = [F * N_to_lbf for F in F_joint_vals]
    N_ftlb = [N * Nm_to_ftlb for N in N_joint_vals]

    # ---------- Report ----------
    print("\n" + "="*72)
    print(f"STATIC LOAD ANALYSIS (NEWTON–EULER) — Configuration (deg): {joint_angles_deg}")
    print("="*72)

    for j in range(3):
        print(f"\nJoint {j+1}:")
        print("  Reaction Force [lbf]:")
        sp.pprint(F_lbf[j])
        print("  Joint Moment [ft·lbf]:")
        sp.pprint(N_ftlb[j])

    print("\nMotor Torques τ [ft·lbf]:")
    sp.pprint(tau_ftlb)
    print("="*72)
