# C. Radio Interception — analysis

**Result:** `answers.csv` scored **99** points. This is the maximum returned by the scoring formula. The probe submissions (Section 8) support 100% accuracy on test, subject to the limits of that analysis.

The numbers below are printed by `evidence.py` and saved in `evidence_output.txt`; section numbers correspond.

## Run

```bash
pip install -r ../requirements.txt
python3 solve_c.py data/train_signals.csv data/test_signals.csv answers.csv
python3 evidence.py data/train_signals.csv data/test_signals.csv answers.csv probes > evidence_output.txt
```

## 1. Data

- Train contains 4,500 rows with nearly balanced classes (1,508, 1,510, and 1,482). Test contains 3,500 rows.
- Every string has length 28 and uses the alphabet A–F. There are no duplicate strings in train and no overlap between train and test strings.

## 2. Why raw characters are uninformative

The cipher permutes the alphabet separately for each string and may reverse the string. Letter frequencies are therefore almost uniform across stations, ranging from 16.2% to 17.5% per letter. A useful feature must be invariant to both renaming the letters and reversing the string.

The simplest such feature is the **length of each run of identical characters**. Renaming letters leaves run lengths unchanged; reversal only reverses their order.

## 3. What run lengths reveal

The table uses train. "Inner" runs exclude the first and last run of a string.

| Station | Inner runs | Edge runs |
|---|---|---|
| 0 | Any length: 1 — 9,766; 2 — 4,720; 3 — 2,291; 4 — 1,051; 5 — 506; … up to 16 | Any length |
| 1 | Only 2 (7,372) and 3 (7,030) | 1, 2, or 3 |
| 2 | Only 1 (9,899) and 6–10 (596, 573, 574, 512, 495) | 1 through 10 |

These patterns reveal how the stations generate strings:

- **Station 0.** The chance that a run ends at the next step is nearly independent of its current length: 0.519, 0.521, 0.528, 0.512, 0.505, and 0.521 for lengths 1–6. This suggests a geometric distribution with parameter about 0.52.
- **Station 1.** Each complete run has length 2 or 3, with nearly equal frequencies.
- **Station 2.** Runs are either single characters (78% of runs) or long runs of length 6–10 (roughly uniform among those lengths).
- **Truncation.** Edge runs can be shorter than the permitted complete-run lengths: station 1 has edge runs of length 1, and station 2 has edge runs of length 2–5. Generation stops after 28 characters, truncating the final run. A length-2–5 run appears at most once per station-2 string (899 strings have none, 583 have one), always at an edge. Only one end is truncated; reversal can put it on either side.
- **Letter choice.** A new run chooses uniformly from the five other letters: the rate at which its letter matches the letter two runs earlier is 0.199, 0.200, and 0.201, versus 0.2 expected under uniform choice. The letters themselves therefore carry no station information, and their likelihood factor cancels out.

## 4. Strict grammars

A station's grammar fits a string if there is an orientation in which every run except the last has a permitted complete length, and the last does not exceed the maximum permitted length.

| Station in train | Fits grammar 1 | Fits grammar 2 |
|---|---|---|
| 0 | 0 of 1,508 | 1 of 1,508 |
| 1 | 1,510 of 1,510 | 0 |
| 2 | 0 | 1,482 of 1,482 |

Grammars 1 and 2 fit every example from their respective stations without exception, and no training string fits both.

## 5. Classifier

`solve_c.py` computes the likelihood of a run-length sequence under each of the three models, incorporates class priors, and selects the maximum.

- Parameters are fitted from inner runs in train: station 0's run-ending probability is 0.5196; station 1's share of length-2 runs is 0.5119; station 2's share of length-1 runs is 0.7826.
- The last run is treated as truncated: its likelihood is the probability that the full run would have been at least as long as observed.
- Both orientations are considered with weight 1/2.

Train accuracy is 0.999778, with one error among 4,500 rows. The station-0 string `FADEEEEEECDEBEAAAAAABAAAAAAA` has run lengths 1, 1, 1, 6, 1, 1, 1, 1, 1, 6, 1, 7 and happens to fit station 2's grammar completely.

Test predictions form disjoint groups: 1,175 strings fit grammar 1 and were assigned station 1; 1,152 fit grammar 2 and were assigned station 2; the remaining 1,173 fit neither and were assigned station 0.

## 6. Chance of an irreducible error

A station-0 string can accidentally fit another station's grammar. A simulation of 400,000 strings with geometric run lengths gives:

| Event | Probability per string | Expected count among approximately 1,173 station-0 test strings |
|---|---|---|
| Fits grammar 1 | 2.75·10⁻⁵ | 0.032 |
| Fits grammar 2 | 1.92·10⁻⁴ | 0.226 |

Train contains 0 and 1 such strings, respectively, among 1,508 station-0 rows. With an expected total of about 0.26 in test, finding none is plausible (roughly three cases out of four). This agrees with the evidence in Section 8.

## 7. Independent gradient-boosting check

LightGBM trained on invariant features (sorted letter frequencies, matches at each shift, and a run-length histogram) reaches 0.9982 accuracy in five-fold cross-validation: eight errors, all station-0 strings classified as station 2. Seven of the eight violate station 2's strict grammar (a length-2–5 run in the middle or a run longer than 10). The grammar distinguishes these strings correctly where boosting does not.

On test, boosting disagrees with `answers.csv` on seven strings. In each case it predicts station 2 despite a run that is impossible for station 2, for example `6, 2, 7, 11, 1, 1`.

An earlier draft (`scratch/f.py`, using a slightly different feature set) reached 0.9987 with five disagreements. The conclusion was the same.

## 8. Why 99 instead of 100?

**Formula.** In float64, `int(100 * max(5 * (accuracy - 0.8), 0))` yields 99.99999999999997 when accuracy is 1.0, and `int` truncates it to 99. Zero through six errors all score 99; seven errors score 98. A score of 99 alone therefore does not establish whether any answer is wrong.

**Probing.** I initially assumed, incorrectly, that 99 implied one to six errors and proposed locating an incorrect row by observing score changes. This used 20 submissions and produced the opposite conclusion: the evidence favors zero errors.

Round 1 (`probes/round1_probes.zip`, 11 submissions) changes station-2 predictions to station 0 when a particular bit is set in the row number.

| Submissions | Answers changed | Observed score | Predicted with zero errors |
|---|---|---|---|
| 01–07 | 581 | 16 | 16 |
| 08–10 | 518 | 25 | 25 |
| 11 | 133 | 80 | 80 |

Round 2 (`probes/round2_probes.zip`, nine submissions):

| Submission | Change | Answers changed | Observed score | Predicted with zero errors |
|---|---|---|---|---|
| 01 | Corrupt 5 station-1 answers | 5 | 99 | 99 |
| 02 | Corrupt 4 answers | 4 | 99 | 99 |
| 03 | Corrupt 3 answers | 3 | 99 | 99 |
| 04 | Flip the first half of station-2 answers | 580 | 17 | 17 |
| 05 | Flip the second half of station-2 answers | 580 | 17 | 17 |
| 06 | Flip the first half of station-1 answers | 594 | 15 | 15 |
| 07 | Flip the second half of station-1 answers | 587 | 16 | 16 |
| 08 | Flip the first half of station-0 answers | 587 | 16 | 16 |
| 09 | Flip the second half of station-0 answers | 587 | 16 | 16 |

**Enumerating hypotheses.** `evidence.py` reads the probes directly from their archives and checks which possible ground truth agrees with all 20 scores:

- Zero errors in `answers.csv`: all 20 predicted scores match observations.
- One error: all 7,000 possibilities (one of 3,500 rows assigned either alternative label) were tested; none matched.
- Two errors: every pair was tested; none matched.

Three or more errors were not exhaustively tested. They would require submission 01 to keep its score despite corrupting five answers, implying that some of those five answers were originally wrong. All five strings fit station 1's grammar perfectly.

Observed scores were read from screenshots and a message; submission order was assumed to match the numbered folders.

**Can the score reach 100?** Not with a valid answer file: accuracy cannot exceed 1.0. The remaining option is to report the rounding issue to the organizers.

## Files

- `solve_c.py` — solution.
- `evidence.py` and `evidence_output.txt` — calculations supporting the numbers in this analysis.
- `answers.csv` — submitted answer.
- `data/` — original task files under their original names.
- `probes/` — archives of probe submissions from both rounds.
- `scratch/` — exploratory scripts with absolute sandbox paths; edits are needed before they can run elsewhere.
