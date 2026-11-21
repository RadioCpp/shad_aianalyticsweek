# A. Mirror for Algernon

> Transcribed from the contest page. The original screenshot is `statement.png` (it contains the histogram).

Researchers studied whether the presence of a mirror is associated with laboratory mice's choice of where to spend their time.

16 laboratory mice were individually placed in cages consisting of two rooms connected by a tunnel. One of the rooms contained a mirror, while the other did not. During the experiment, the position of the mirror was changed to reduce the influence of room location on the results.

For each mouse, the proportion of time spent in the room with the mirror was determined. The distribution of the resulting values is shown in the histogram.

![histogram](statement.png)

Histogram as read from the picture (x axis: "Proportion of time in cage containing mirror (%)", from 30 to 70; y axis: "Number of mice", from 0 to 10):

| Bin, % | Number of mice |
|---|---|
| 30–35 | 1 |
| 40–45 | 3 |
| 45–50 | 9 |
| 50–55 | 2 |
| 60–65 | 1 |

The horizontal axis shows the proportion of time spent in the room containing mirror, expressed as a percentage; the vertical axis shows the number of mice.

It is known that each mouse spent more time in one of the two rooms than in the other. Assume that the results for different mice are independent and that the probability of spending more time in the room without a mirror is the same for all mice.

Can we conclude at the 5% significance level that mice tend to spend more time in one of the two rooms --- with or without a mirror?

Propose a statistical test based on the available data, calculate the exact two-sided p-value, and draw a conclusion at the significance level α = 0.05. When calculating the p-value, account for deviations from the expected result in both the greater and the lesser directions.

## Output format

In the answer field, enter two numbers separated by a space:

- the p-value of the statistical test;
- the number 1 if the null hypothesis is rejected at the significance level α = 0.05, or the number 0 otherwise.

The absolute error of the p-value must not exceed 10^-4.
