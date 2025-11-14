# Kai De La Cruz
# 3DOF Leg Static Load Analysis using Newton–Euler Method
# Feature: Sweep θ₁–θ₂–θ₃ and report, for EACH joint:
#          - max reaction force magnitude + vector (lbf)
#          - joint moment vector at that configuration (ft·lbf)  
#          - motor torques τ at that configuration (ft·lbf)      

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# ---------- DH Transform ----------
def dh_transform_sym(theta, d, a, alpha):
    ct, st = sp.cos(theta), sp.sin(theta)
    ca, sa = sp.cos(alpha), sp.sin(alpha)
    T = sp.Matrix([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0,      sa,     ca,   d],
        [0,       0,      0,   1]
    ])
    return sp.simplify(T)

# ---------- Frame Transformation ----------
def frame_tf(Tn, Tm):
    return Tm @ Tn

# ---------- Extract z & p ----------
def output_z_p(T):
    return T[0:3, 2], T[0:3, 3]

# ---------- Newton–Euler Statics ----------
def NE_statics(q_syms, L_syms, T_mats, masses, g_vec, F_ext):
    """Static recursion returning torques + reaction forces/moments at each joint (base frame)."""
    # Forward kinematics
    T02 = frame_tf(T_mats[0], T_mats[1])
    T03 = frame_tf(T02, T_mats[2])
    T0e = frame_tf(T03, T_mats[3])

    z1, p1 = output_z_p(T_mats[0])
    z2, p2 = output_z_p(T02)
    z3, p3 = output_z_p(T03)
    _, pe = output_z_p(T0e)
    z, p = [z1, z2, z3], [p1, p2, p3]

    # COMs (approx; replace with real values if you have them)
    pC1, pC2, pC3 = p1/2, (p1+p2)/2, (p2+pe)/2

    F_next = -F_ext
    N_next = sp.Matrix([0, 0, 0])
    tau, joint_forces, joint_moments = [], [], []

    for i in reversed(range(3)):
        mi = masses[i]
        F_i = F_next + mi*g_vec
        if i == 2:
            r_next = pe - p3
        elif i == 1:
            r_next = p3 - p2
        else:
            r_next = p2 - p1
        N_i = N_next + r_next.cross(F_next) + (pC1 if i==0 else pC2 if i==1 else pC3).cross(mi*g_vec)
        tau_i = z[i].dot(N_i)
        tau.insert(0, sp.simplify(tau_i))
        joint_forces.insert(0, sp.simplify(F_i))
        joint_moments.insert(0, sp.simplify(N_i))
        F_next, N_next = F_i, N_i

    tau_vec = -sp.Matrix(tau)  # actuator torques resisting external load
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
    F_ext = sp.Matrix([0, 0, 58])  # N upward
    masses = [0.5, 0.4, 0.3]
    subs_base = {L1: 0.061, L2: 0.083, L3: 0.146, L4: 0.165, g: 9.81}

    # Angle sweep ranges (deg)
    theta1_range = np.linspace(-45, 45, 5)
    theta2_range = np.linspace(0, 90, 5)
    theta3_range = np.linspace(0, 135, 5)

    # Tracking per-joint maxima
    max_force_mag = [0, 0, 0]
    best_config = [(0,0,0)]*3
    best_force_vec = [None]*3
    best_moment_vec = [None]*3      
    best_tau_vec = [None]*3          

    print("Sweeping θ₁–θ₂–θ₃ space...")

    for t1 in theta1_range:
        for t2 in theta2_range:
            for t3 in theta3_range:
                subs = subs_base.copy()
                subs.update({
                    theta1: np.deg2rad(t1),
                    theta2: np.deg2rad(t2),
                    theta3: np.deg2rad(t3),
                    m1: 0.5, m2: 0.4, m3: 0.3
                })

                tau_sym, F_list, N_list = NE_statics(
                    [theta1, theta2, theta3],
                    [L1, L2, L3, L4],
                    [T01, T12, T23, T3e],
                    masses,
                    g_vec,
                    F_ext
                )

                # Evaluate numeric values at this configuration
                F_joint_vals = [F.evalf(subs=subs) for F in F_list]
                N_joint_vals = [N.evalf(subs=subs) for N in N_list]   
                tau_vals = tau_sym.evalf(subs=subs)                   

                # Magnitudes of forces per joint
                F_mags = [float(sp.sqrt(F.dot(F))) for F in F_joint_vals]

                for j in range(3):
                    if F_mags[j] > max_force_mag[j]:
                        max_force_mag[j] = F_mags[j]
                        best_config[j]   = (t1, t2, t3)
                        best_force_vec[j]= F_joint_vals[j]
                        best_moment_vec[j]= N_joint_vals[j]   
                        best_tau_vec[j]   = tau_vals          

    # ---------- Imperial conversions ----------
    N_to_lbf   = 0.224809
    Nm_to_ftlb = 0.73756214927727

    max_force_lbf  = [m * N_to_lbf for m in max_force_mag]
    best_force_lbf = [F * N_to_lbf for F in best_force_vec]
    best_moment_ftlb = [M * Nm_to_ftlb for M in best_moment_vec]  
    best_tau_ftlb    = [tau * Nm_to_ftlb for tau in best_tau_vec] 

    # ---------- Report ----------
    print("\n" + "="*72)
    print("STATIC LOAD ANALYSIS (NEWTON–EULER)")
    print("Maximum Reaction Force Magnitude per Joint — with Moments & Torques")
    print("="*72)

    for j in range(3):
        print(f"\nJoint {j+1}:")
        print(f"  Max Reaction Force: {max_force_mag[j]:.2f} N  ({max_force_lbf[j]:.2f} lbf)")
        print(f"  Occurs at configuration (deg): "
              f"θ₁={best_config[j][0]:.1f}, θ₂={best_config[j][1]:.1f}, θ₃={best_config[j][2]:.1f}")

        print("  Reaction Force Vector [lbf]:")
        sp.pprint(best_force_lbf[j])

        print("  Joint Moment Vector about Origin [ft·lbf]:")     
        sp.pprint(best_moment_ftlb[j])

        print("  Motor Torques τ = [τ1, τ2, τ3] [ft·lbf]:")       
        sp.pprint(best_tau_ftlb[j])

    print("\n" + "="*72)

