"""Minimum detectable absolute risk difference for the primary contrast.

Fixed-cohort precision statement: two-sided alpha 0.05, 80% power, 1563 episodes
per arm, observed shallow-arm risk as the reference proportion.
"""
import math
from scipy.stats import norm
from scipy.optimize import brentq

N = 1563
P1 = 0.0883728965382424  # observed shallow-arm 7-day risk
ALPHA, POWER = 0.05, 0.80
za, zb = norm.ppf(1 - ALPHA / 2), norm.ppf(POWER)


def power_at(p2):
    pbar = (P1 + p2) / 2
    num = abs(p2 - P1) * math.sqrt(N)
    den_null = math.sqrt(2 * pbar * (1 - pbar))
    den_alt = math.sqrt(P1 * (1 - P1) + p2 * (1 - p2))
    return norm.cdf((num - za * den_null) / den_alt)


p2 = brentq(lambda p: power_at(p) - POWER, P1 + 1e-6, 0.5)
print(f"Reference (shallow) risk: {P1*100:.1f}%")
print(f"Per-arm n: {N}")
print(f"Minimum detectable risk: {p2*100:.2f}%")
print(f"Minimum detectable absolute difference: {(p2-P1)*100:.2f} percentage points")
print(f"Corresponding risk ratio: {p2/P1:.2f}")
print(f"Achieved power at the observed RD of 5.1 points: {power_at(P1+0.0512)*100:.1f}%")
