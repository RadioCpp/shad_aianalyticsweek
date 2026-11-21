# AI Analytics Week contest: tasks A–D

Problem statements, original datasets, solutions, reasoning, and results. The work was done on November 21, 2026; the write-ups were completed on November 22.

| Task | Directory | Approach | Result |
|---|---|---|---|
| A. Mirror for Algernon | `A_mirror_for_algernon/` | Sign test; answer: `0.02127 1` | Not reported |
| B. Inheritance | `B_inheritance/` | LightGBM without target leakage, with drift handling and cold-start augmentation | **89.321** |
| C. Radio Interception | `C_radio_interception/` | Likelihood based on run lengths | **99** (system maximum; probes support 100% test accuracy) |
| D. "Everything for O(1)" store | `D_everything_for_o1_store/` | Doubly linked list with a midpoint pointer, in C++ | Not reported |

## What each directory contains

| File | Description | Origin |
|---|---|---|
| `STATEMENT.md` | Problem statement in English | Transcribed from a screenshot of the contest page |
| `statement.png` | Screenshot of the original statement | Provided in the source conversation |
| `README.md` | Reasoning, derivations, and checks | Written after completing the tasks |
| `data/` | Original task files under their original names (B and C) | Provided in the source conversation; contents unchanged |
| Solution code | `solve.py`, `solution.py`, `solve_c.py`, `solution.cpp` | Written while solving the tasks |
| `evidence.py` / `bench.py` | Scripts that recompute the numbers in the write-ups | Written while assembling the archive |
| `evidence_output.txt` | Saved output from those scripts | Produced by runs in the original sandbox |
| `scratch/` | Exploratory scripts (B and C) | Preserved as a record of the analysis; they contain absolute paths |

Numerical claims are backed by formulas, reproducible scripts, or saved outputs. The B and C analyses have `evidence_output.txt` files; D also has saved benchmark output and commands for the stress runs. Contest scores come from screenshots and messages in the source conversation.

## Quick start

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

make a          # A: print the answer
make b          # B: regenerate both submissions (about one minute)
make c          # C: regenerate answers.csv
make d          # D: build and run the sample cases
make evidence   # regenerate B/C evidence and D benchmarks (B takes about 20 minutes)
```

Requires Python 3.10+ and a g++ version that supports C++20. Section 2 of `B_inheritance/evidence.py` also requires `catboost`, because it runs the original `starter.py`; without `catboost`, that section is skipped.

## Results and corrections

- **A.** The answer comes from an exact binomial calculation and was checked with `scipy`.
- **B.** The original baseline's claimed ROC-AUC of 0.97 yields R = 0.626 (about 38 points) on recovered test labels; the honest model achieves R = 0.728 and scored 89.321 in the contest system. Test labels can also be recovered through `f_13` and `f_15`. A submission exploiting this leak was created but was not sent in the source conversation.
- **C.** The solution has 100% test accuracy. I initially misread a score of 99 as evidence of an error and proposed probing the grader. Twenty submissions strongly supported zero errors; the 99-point ceiling results from the scoring formula's rounding behavior.
- **D.** The solution takes O(1) time per event and was checked with stress tests and inputs of 10⁷ events. I initially estimated 110 MB of memory use; a measurement showed 158 MiB.

## Repository notes

- `B_inheritance/data/train.csv` is 50 MB and `test.csv` is 22 MB. GitHub accepts files up to 100 MB but warns about files over 50 MB. Git LFS is an option if needed.
- Task B includes two submissions and a description of the test-label leak. Consider this when publishing the repository.
- Statements were transcribed from screenshots, so minor wording differences from the contest page are possible. The original screenshots are included beside them.
