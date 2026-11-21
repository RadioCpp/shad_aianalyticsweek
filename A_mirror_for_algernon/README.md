# A. Mirror for Algernon — analysis

**Answer:** `0.02127 1`

The contest verdict for this task was not reported in the source conversation.

## 1. Available data

The statement (`STATEMENT.md`) provides a histogram for 16 mice, not their individual proportions of time. Exact values within the bins are unknown. The only reliable information we can extract is which side of 50% each mouse falls on.

Reading the histogram (`statement.png`):

- The X-axis runs from 30 to 70, with labels every 10 units and intermediate ticks every 5. Each bar spans one five-percentage-point interval.
- Bar heights are 1 (30–35), 3 (40–45), 9 (45–50), 2 (50–55), and 1 (60–65).
- As a check, 1 + 3 + 9 + 2 + 1 = 16, matching the number of mice in the statement.

| Side of 50% | Bins | Mice |
|---|---|---|
| Less than 50% of time with the mirror (preferred the room without it) | 30–35, 40–45, 45–50 | 13 |
| More than 50% of time with the mirror (preferred the room with it) | 50–55, 60–65 | 3 |

The 50% boundary creates no ambiguity: the statement says every mouse spent more time in one room than in the other, so no observation is exactly 50%. The 45–50 bin lies entirely to the left of the boundary, and 50–55 lies to the right.

## 2. Why a sign test?

The statement itself specifies the model:

- "each mouse spent more time in one of the two rooms" — each mouse has a binary outcome;
- "results for different mice are independent" — the trials are independent;
- "the probability of spending more time in the room without a mirror is the same for all mice" — the success probability is shared.

These are Bernoulli trials, so the number of mice preferring the room with the mirror is binomially distributed. Under the null hypothesis that the mirror does not affect preference, p = 1/2. Testing that hypothesis using only the counts of preferences is a sign test (an exact binomial test).

Why not another test?

- A **t-test** needs individual values and their variance. Five-percentage-point histogram bins give only approximations.
- A **Wilcoxon signed-rank test** needs ranks of deviations from 50%. The ranks are unknown within a bin, such as the bin containing nine mice at 45–50%.
- The problem asks for an **exact** p-value, which follows naturally from the discrete binomial distribution.

## 3. Calculation

Let X be the number of mice preferring the room with the mirror. Under H₀, X ~ Binomial(16, 1/2); the observed value is X = 3.

"Deviations in both directions" means outcomes at least as far from the expected value of 8 as the observation: X ≤ 3 or X ≥ 13. By symmetry, the two tails have equal probabilities.

| k | C(16, k) |
|---|---|
| 0 | 1 |
| 1 | 16 |
| 2 | 120 |
| 3 | 560 |
| Sum | 697 |

- P(X ≤ 3) = 697 / 2¹⁶ = 697 / 65536 ≈ 0.0106354
- p = 2 · 697 / 65536 = 697 / 32768 ≈ **0.0212708**

The alternative definition of a two-sided p-value as the sum of probabilities of outcomes no more likely than the observed one gives the same result, because the distribution is symmetric when p = 1/2.

## 4. Conclusion

Since 0.0212708 < 0.05, we reject the null hypothesis: the mice prefer one of the rooms (here, the room without the mirror, preferred by 13 of 16 mice). The second number in the answer is 1.

The required tolerance is 10⁻⁴; `0.02127` differs from the exact value by about 8·10⁻⁷.

## 5. Checks

- `solve.py` calculates the p-value using exact fractions: `697/32768 = 0.0212708`.
- `scipy.stats.binomtest(3, 16, 0.5)` gives 0.021270751953125 (the one-sided value is 0.0106353759765625).
- Sensitivity to reading the histogram: a 12-to-4 split would give p ≈ 0.0768 (no rejection of H₀), while a 14-to-2 split would give p ≈ 0.0042. The conclusion depends on there being exactly three mice in bins to the right of 50%, as supported by the bar heights summing to 16.

## Run

```bash
python3 solve.py
```

Only the Python 3 standard library is required.
