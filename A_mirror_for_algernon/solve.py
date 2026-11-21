#!/usr/bin/env python3
"""A. Mirror for Algernon -- exact two-sided sign test.

Histogram (read from statement.png): bins 30-35: 1, 40-45: 3, 45-50: 9, 50-55: 2, 60-65: 1.
No mouse sits at exactly 50% (each one preferred a room), so
13 mice preferred the room without the mirror and 3 the room with it.
Under H0 the number of mice preferring the mirror room is Binomial(16, 0.5).
"""
from fractions import Fraction
from math import comb

BINS = {(30, 35): 1, (40, 45): 3, (45, 50): 9, (50, 55): 2, (60, 65): 1}
ALPHA = 0.05

n = sum(BINS.values())
mirror = sum(c for (lo, hi), c in BINS.items() if lo >= 50)
no_mirror = n - mirror
k = min(mirror, no_mirror)

# symmetric null distribution: both tails are equal
p = Fraction(2 * sum(comb(n, i) for i in range(k + 1)), 2 ** n)
p = min(p, Fraction(1))

if __name__ == "__main__":
    print(f"n = {n}, mirror = {mirror}, no mirror = {no_mirror}")
    print(f"p-value = {p.numerator}/{p.denominator} = {float(p):.7f}")
    print(f"answer: {float(p):.5f} {int(float(p) < ALPHA)}")
