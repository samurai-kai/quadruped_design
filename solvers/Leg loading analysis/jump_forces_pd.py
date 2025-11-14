#!/usr/bin/env python3
"""
jump_forces_pd.py — Lift-off & impact force estimator with PD-derived landing time

Two ways to set landing dynamics:
1) Direct time-limited landing (same as simple script):
   --land 0.06
2) PD landing (preferred when you run PD/impedance on legs):
   --kp 15000 --kd 500  (per-leg vertical gains, N/m and N*s/m)
   The script sums gains over stance legs to get K_tot, D_tot, computes
     ω_n = sqrt(K_tot/m), ζ = D_tot / (2*sqrt(m*K_tot)),
   and uses   t_landing ≈ 3 / (ζ * ω_n)   (well-damped, fast-stop approximation).

Examples:
  python jump_forces_pd.py --mass 11.829 --height 0.53 --stance 0.15 --legs 4 --kp 15000 --kd 500
  python jump_forces_pd.py --mass 11.829 --height 0.30 --stance 0.2 --land 0.1 --legs 4

Outputs a summary and two plots (takeoff & landing GRF).
"""

from dataclasses import dataclass, asdict
from math import sqrt
from typing import Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt

GRAVITY = 9.81

@dataclass
class JumpParams:
    mass: float = 11.829               # [kg] total robot mass
    height: float = 0.50               # [m] apex height above takeoff
    stance_time: float = 0.15          # [s] takeoff push duration
    legs_in_stance: int = 4            # [-] legs sharing load
    # Takeoff/landing force profile shape
    takeoff_peak_ratio: float = 2.0    # [-] peak/avg during takeoff
    landing_peak_ratio: float = 2.0    # [-] peak/avg during landing
    rise_frac: float = 0.3             # [-] trapezoid rise fraction
    fall_frac: float = 0.3             # [-] trapezoid fall fraction
    # Landing time choice (either direct or from PD)
    landing_time_direct: Optional[float] = None  # [s] if provided, overrides PD
    kp_per_leg: Optional[float] = None # [N/m] per-leg vertical proportional gain
    kd_per_leg: Optional[float] = None # [N*s/m] per-leg vertical derivative gain
    g: float = GRAVITY                 # gravity [m/s^2]

@dataclass
class JumpResults:
    v0: float
    J_takeoff: float
    Fgrf_avg_takeoff: float
    Fgrf_peak_takeoff: float
    Fleg_avg_takeoff: float
    Fleg_peak_takeoff: float

    v_impact: float
    landing_time_used: float
    zeta: Optional[float]
    omega_n: Optional[float]
    Fgrf_avg_land_time: float
    Fgrf_peak_land_time: float
    Fleg_peak_land_time: float

def trapezoid_profile(T: float, Favg: float, Fpeak: float, mg: float,
                      rise_frac: float, fall_frac: float, n: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    plateau_frac = max(0.0, 1.0 - rise_frac - fall_frac)
    t = np.linspace(0.0, T, n)
    if T <= 0.0:
        return t, np.full_like(t, Favg)

    t1 = rise_frac * T
    t2 = t1 + plateau_frac * T
    F = np.zeros_like(t)

    for i, ti in enumerate(t):
        if ti <= max(t1, 1e-9):
            # ramp from mg to peak
            F[i] = mg + (Fpeak - mg) * (ti / max(t1, 1e-9))
        elif ti <= t2:
            F[i] = Fpeak
        else:
            # ramp from peak to mg
            denom = max(T - t2, 1e-9)
            F[i] = Fpeak - (Fpeak - mg) * ((ti - t2) / denom)

    # Scale to match desired average exactly
    current_avg = F.mean() if F.size else 1.0
    if current_avg > 0:
        F *= (Favg / current_avg)
    return t, F

def compute_pd_landing_time(mass: float, legs: int, kp_leg: float, kd_leg: float) -> Tuple[float, float, float]:
    """
    Returns (t_landing, zeta, omega_n) from per-leg vertical gains.
    t_landing ≈ 3 / (ζ * ω_n). Clipped to a small positive minimum.
    """
    legs = max(int(legs), 1)
    K_tot = kp_leg * legs
    D_tot = kd_leg * legs
    if K_tot <= 0.0:
        raise ValueError("kp must be > 0 to compute PD landing time.")
    omega_n = sqrt(K_tot / mass)
    zeta = D_tot / (2.0 * sqrt(mass * K_tot))
    # Avoid degenerate small ζ or ωn -> large landing time; clip ζ to small positive
    zeta_eff = max(zeta, 1e-3)
    omega_eff = max(omega_n, 1e-6)
    t_landing = 3.0 / (zeta_eff * omega_eff)
    # Numerical lower/upper bounds to avoid extremes
    t_landing = float(np.clip(t_landing, 1e-3, 1.0))
    return t_landing, zeta, omega_n

def compute_forces(p: JumpParams) -> JumpResults:
    m, g = p.mass, p.g
    v0 = sqrt(2 * g * p.height)
    J_takeoff = m * v0

    Fnet_avg_takeoff = J_takeoff / max(p.stance_time, 1e-9)
    Fgrf_avg_takeoff = Fnet_avg_takeoff + m * g
    Fgrf_peak_takeoff = p.takeoff_peak_ratio * Fgrf_avg_takeoff
    Fleg_avg_takeoff = Fgrf_avg_takeoff / max(p.legs_in_stance, 1)
    Fleg_peak_takeoff = Fgrf_peak_takeoff / max(p.legs_in_stance, 1)

    v_impact = v0  # symmetric ballistic

    # Decide landing time: direct or PD-derived
    zeta = omega_n = None
    if p.landing_time_direct is not None:
        t_land = float(p.landing_time_direct)
    elif (p.kp_per_leg is not None) and (p.kd_per_leg is not None):
        t_land, zeta, omega_n = compute_pd_landing_time(m, p.legs_in_stance, p.kp_per_leg, p.kd_per_leg)
    else:
        raise ValueError("Specify either --land (landing_time_direct) or both --kp and --kd for PD-derived landing time.")

    # Time-limited landing model
    J_landing = m * v_impact
    Fnet_avg_land_time = J_landing / max(t_land, 1e-9)
    Fgrf_avg_land_time = Fnet_avg_land_time + m * g
    Fgrf_peak_land_time = p.landing_peak_ratio * Fgrf_avg_land_time
    Fleg_peak_land_time = Fgrf_peak_land_time / max(p.legs_in_stance, 1)

    return JumpResults(
        v0=v0,
        J_takeoff=J_takeoff,
        Fgrf_avg_takeoff=Fgrf_avg_takeoff,
        Fgrf_peak_takeoff=Fgrf_peak_takeoff,
        Fleg_avg_takeoff=Fleg_avg_takeoff,
        Fleg_peak_takeoff=Fleg_peak_takeoff,
        v_impact=v_impact,
        landing_time_used=t_land,
        zeta=zeta,
        omega_n=omega_n,
        Fgrf_avg_land_time=Fgrf_avg_land_time,
        Fgrf_peak_land_time=Fgrf_peak_land_time,
        Fleg_peak_land_time=Fleg_peak_land_time,
    )

def print_summary(p: JumpParams, r: JumpResults) -> None:
    # --- unit factors ---
    N2lbf   = 0.224809
    m2ft    = 3.28084
    kg2lbm  = 2.20462

    # Body weight in N and lbf
    BW_N   = p.mass * p.g
    BW_lbf = BW_N * N2lbf

    def row(name: str, val: float, unit: str = ""):
        unit = f" {unit}" if unit else ""
        print(f"{name:<40s}: {val:>10.3f}{unit}")

    # ---- Inputs (Imperial) ----
    print("\nInputs (Imperial):")
    # Common fields
    print(f"{'mass':>22s}: {p.mass * kg2lbm:.3f} lbm")
    print(f"{'height':>22s}: {p.height * m2ft:.3f} ft")
    # Use landing_time_direct if present (PD variant); otherwise skip
    if hasattr(p, 'stance_time'):
        print(f"{'stance_time':>22s}: {p.stance_time:.3f} s")
    if hasattr(p, 'landing_time_direct') and (p.landing_time_direct is not None):
        print(f"{'landing_time (direct)':>22s}: {p.landing_time_direct:.3f} s")
    if hasattr(p, 'legs_in_stance'):
        print(f"{'legs_in_stance':>22s}: {p.legs_in_stance}")
    if hasattr(p, 'takeoff_peak_ratio'):
        print(f"{'takeoff_peak_ratio':>22s}: {p.takeoff_peak_ratio}")
    if hasattr(p, 'landing_peak_ratio'):
        print(f"{'landing_peak_ratio':>22s}: {p.landing_peak_ratio}")
    if hasattr(p, 'rise_frac'):
        print(f"{'rise_frac':>22s}: {p.rise_frac}")
    if hasattr(p, 'fall_frac'):
        print(f"{'fall_frac':>22s}: {p.fall_frac}")
    # Optional PD gains (convert N/m -> lbf/ft, N*s/m -> lbf*s/ft)
    if hasattr(p, 'kp_per_leg') and p.kp_per_leg is not None:
        print(f"{'kp_per_leg':>22s}: {p.kp_per_leg * (N2lbf / m2ft):.3f} lbf/ft")
    if hasattr(p, 'kd_per_leg') and p.kd_per_leg is not None:
        print(f"{'kd_per_leg':>22s}: {p.kd_per_leg * (N2lbf / m2ft):.3f} lbf*s/ft")

    # ---- Outputs (Imperial) ----
    print("\nOutputs (Imperial):")
    row("Takeoff required v0", r.v0 * m2ft, "ft/s")
    row("Takeoff impulse J",   r.J_takeoff * N2lbf, "lbf*s")
    row("Avg GRF (takeoff)",   r.Fgrf_avg_takeoff * N2lbf, "lbf")
    row("Peak GRF (takeoff)",  r.Fgrf_peak_takeoff * N2lbf, "lbf")
    row("Per-leg avg (takeoff)",  r.Fleg_avg_takeoff * N2lbf, "lbf")
    row("Per-leg peak (takeoff)", r.Fleg_peak_takeoff * N2lbf, "lbf")

    row("Impact speed (landing)", r.v_impact * m2ft, "ft/s")

    # Landing time (from PD or direct)
    if hasattr(r, 'landing_time_used'):
        row("Landing time used", r.landing_time_used, "s")

    # Optional PD diagnostics
    if hasattr(r, 'zeta') and hasattr(r, 'omega_n') and (r.zeta is not None) and (r.omega_n is not None):
        row("zeta (damping ratio)", r.zeta, "-")
        row("omega_n (natural freq.)", r.omega_n, "rad/s")

    row("Avg GRF (landing, time)",  r.Fgrf_avg_land_time * N2lbf, "lbf")
    row("Peak GRF (landing, time)", r.Fgrf_peak_land_time * N2lbf, "lbf")
    row("Per-leg peak (landing, time)", r.Fleg_peak_land_time * N2lbf, "lbf")

    # BW-normalized peaks (dimensionless)
    print("\nBW-normalized (dimensionless):")
    row("Peak GRF (takeoff) / BW",  (r.Fgrf_peak_takeoff * N2lbf) / BW_lbf, "x")
    row("Peak GRF (landing) / BW",  (r.Fgrf_peak_land_time * N2lbf) / BW_lbf, "x")
    print()

def plot_profiles(p: JumpParams, r: JumpResults) -> None:
    """
    Plot takeoff and landing GRF profiles, showing both average and peak values
    (in Newtons and lbf) for total vertical ground reaction force.
    """

    import matplotlib.pyplot as plt
    import numpy as np

    # Unit conversions
    N2lbf = 0.224809
    mg = p.mass * p.g

    # Determine landing time (PD-derived or direct)
    T_land = getattr(r, "landing_time_used", None)
    if T_land is None:
        raise ValueError("Landing time not available (need r.landing_time_used).")

    # ---- Generate trapezoidal profiles ----
    # Takeoff
    t_to, F_to = trapezoid_profile(
        p.stance_time,
        r.Fgrf_avg_takeoff,
        r.Fgrf_peak_takeoff,
        mg,
        p.rise_frac,
        p.fall_frac
    )
    # Landing
    t_ld, F_ld = trapezoid_profile(
        T_land,
        r.Fgrf_avg_land_time,
        r.Fgrf_peak_land_time,
        mg,
        p.rise_frac,
        p.fall_frac
    )

    # ---- Helper for plotting ----
    def plot_force_profile(t, F, F_avg, F_peak, title, ylabel="GRF [N]"):
        plt.plot(t, F, label="GRF profile", color="C0", lw=2)
        plt.axhline(F_avg, linestyle="--", color="gray", label=f"Average = {F_avg:.0f} lbf")
        plt.axhline(F_peak, linestyle=":", color="red", label=f"Peak = {F_peak:.0f} lbf")
        plt.scatter([t[np.argmax(F)]], [F_peak], color="red", zorder=3)
        plt.xlabel("Time [s]")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)

    # ---- SI (Newtons) plots ----
    plt.figure(figsize=(6, 4))
    plot_force_profile(
        t_to,
        F_to,
        r.Fgrf_avg_takeoff,
        r.Fgrf_peak_takeoff,
        f"Takeoff Force Profile (Peak/Avg = {r.Fgrf_peak_takeoff / r.Fgrf_avg_takeoff:.2f}×)"
    )

    plt.figure(figsize=(6, 4))
    title_ld = "Landing Force Profile ("
    title_ld += "PD-derived t)" if (p.kp_per_leg and p.kd_per_leg and p.landing_time_direct is None) else "direct t)"
    plot_force_profile(
        t_ld,
        F_ld,
        r.Fgrf_avg_land_time,
        r.Fgrf_peak_land_time,
        f"{title_ld} (Peak/Avg = {r.Fgrf_peak_land_time / r.Fgrf_avg_land_time:.2f}×)"
    )

    # ---- Imperial (lbf) plots ----
    F_to_lbf = F_to * N2lbf
    F_ld_lbf = F_ld * N2lbf
    Favg_to_lbf = r.Fgrf_avg_takeoff * N2lbf
    Fpeak_to_lbf = r.Fgrf_peak_takeoff * N2lbf
    Favg_ld_lbf = r.Fgrf_avg_land_time * N2lbf
    Fpeak_ld_lbf = r.Fgrf_peak_land_time * N2lbf

    plt.figure(figsize=(6, 4))
    plot_force_profile(
        t_to,
        F_to_lbf,
        Favg_to_lbf,
        Fpeak_to_lbf,
        f"Takeoff Force Profile (Peak/Avg = {Fpeak_to_lbf / Favg_to_lbf:.2f}×)",
        ylabel="GRF [lbf]"
    )

    plt.figure(figsize=(6, 4))
    title_ld_imp = "Landing Force Profile ("
    title_ld_imp += "PD-derived t)" if (p.kp_per_leg and p.kd_per_leg and p.landing_time_direct is None) else "direct t)"
    plot_force_profile(
        t_ld,
        F_ld_lbf,
        Favg_ld_lbf,
        Fpeak_ld_lbf,
        f"{title_ld_imp} (Peak/Avg = {Fpeak_ld_lbf / Favg_ld_lbf:.2f}×)",
        ylabel="GRF [lbf]"
    )

    plt.tight_layout()
    plt.show()


def _parse_args():
    import argparse
    ap = argparse.ArgumentParser(description="Jump lift-off and landing force estimator with PD-derived landing time")
    ap.add_argument("--mass", type=float, default=11.829, help="Robot mass [kg]")
    ap.add_argument("--height", type=float, default=0.50, help="Desired apex height [m]")
    ap.add_argument("--stance", type=float, default=0.15, help="Takeoff stance/push time [s]")
    ap.add_argument("--legs", type=int, default=4, help="Number of legs sharing load")
    ap.add_argument("--peak_to", type=float, default=2.0, help="Takeoff peak/avg shape factor")
    ap.add_argument("--peak_ld", type=float, default=2.0, help="Landing peak/avg shape factor")
    ap.add_argument("--rise", type=float, default=0.3, help="Trapezoid rise fraction (0–1)")
    ap.add_argument("--fall", type=float, default=0.3, help="Trapezoid fall fraction (0–1)")
    # Landing time: either direct or PD
    ap.add_argument("--land", type=float, default=None, help="Landing contact time to stop [s] (direct, overrides PD)")
    ap.add_argument("--kp", type=float, default=None, help="Per-leg vertical Kp [N/m]")
    ap.add_argument("--kd", type=float, default=None, help="Per-leg vertical Kd [N*s/m]")
    ap.add_argument("--no-plots", action="store_true", help="Compute & print only (no matplotlib windows)")
    return ap.parse_args()

def main():
    args = _parse_args()
    p = JumpParams(
        mass=args.mass, height=args.height, stance_time=args.stance,
        legs_in_stance=args.legs,
        takeoff_peak_ratio=args.peak_to, landing_peak_ratio=args.peak_ld,
        rise_frac=args.rise, fall_frac=args.fall,
        landing_time_direct=args.land,
        kp_per_leg=args.kp, kd_per_leg=args.kd,
    )
    res = compute_forces(p)
    print_summary(p, res)
    if not args.no_plots:
        plot_profiles(p, res)

if __name__ == "__main__":
    main()
