# B. Inheritance

> Transcribed from the contest page. The original screenshot is `statement.png`.

You have been given a ready-made binary classification pipeline. The attached baseline reports **ROC-AUC ≈ 0.97** on its validation set.

Each row corresponds to an event involving a certain entity at the time specified by `event_time`. In production, the model is applied to **later events**. Among the future data, there are both known entities and groups, as well as those that have not been observed before.

The numerical features `f_00`–`f_23` were obtained using the existing feature pipeline. Its documentation is incomplete, so the correctness of individual features and the original validation protocol cannot be guaranteed.

Your goal is to achieve the highest possible score on the hidden future period. You are allowed to change the validation, the set of features, preprocessing, the model, and the training method.

## Files

- `train.csv` — features and `target`;
- `test.csv` — a later period without `target`;
- `starter.py` — the original baseline;
- `sample_submission.csv` — a sample submission;
- `requirements.txt` — minimal environment.

Main columns:

- `row_id` — row identifier;
- `event_time` — event time in arbitrary seconds;
- `entity_id`, `group_id` — recurring categorical identifiers;
- `cat_00`–`cat_04` — categorical features;
- `f_00`–`f_23` — anonymous numerical features;
- `target` — binary target variable, available only in train.

Rows in `train.csv` are shuffled. Rows in `test.csv` belong to a period subsequent to train.

## Submission format

You must submit **only a CSV file**:

```
row_id,prediction
10000001,0.031
10000002,0.772
```

Requirements:

- exactly one row for each `row_id` from `test.csv`;
- columns must be named `row_id` and `prediction`;
- `prediction` must be a finite number between 0 and 1;
- extra, missing, or duplicate `row_id` values are not allowed.

No text answers or code need to be submitted.

## Metric

The hidden test is divided into three consecutive time blocks. Let `A1`, `A2`, `A3` denote the ROC-AUC on these blocks.

A cold-start subset is defined separately: rows where `entity_id` **or** `group_id` did not appear in train. Let `Acold` denote the ROC-AUC on this subset.

First, robust quality is calculated:

```
R = 0.50 * mean(A1, A2, A3)
  + 0.25 * Acold
  + 0.25 * min(A1, A2, A3, Acold)
```

Final score:

```
Score = 100 * clip((R - 0.55) / 0.20, 0, 1)
```

Benchmarks:

- `0` points: `R ≤ 0.55`;
- `50` points: `R = 0.65`;
- `75` points: `R = 0.70`;
- `100` points: `R ≥ 0.75`.

The higher the score, the better. The best submission counts.

## Answer

Language: (make) yandexdataschool. Submission by file upload; 100 attempts.
