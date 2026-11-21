# C. Radio Interception

> Transcribed from the contest page. The original screenshot is `statement.png`.

Reconnaissance intercepts encrypted radio transmissions from three enemy stations. Each transmission is a string of 28 characters from the alphabet A–F. The problem is that each station encrypts its messages using its own one-time cipher: the alphabet characters are arbitrarily permuted (each transmission has its own permutation), and the message itself may be written in either forward or reverse order. As a result, "raw" characters are useless for identifying the station.

Nevertheless, analysts are confident that each station has its own distinctive transmission "signature" that the cipher cannot completely conceal. Some of the intercepted transmissions have already been attributed to their source stations by experts. Help the intelligence service: use the labeled intercepts to learn how to identify the source station of the remaining transmissions.

## About the Dataset

You are provided with three files: `train_signals.csv`, `test_signals.csv`, and `example.csv`:

- The `train_signals.csv` file contains a `string` column with 28-character strings from the alphabet `A–F` and a `source` column with values `0`, `1`, or `2` — the number of the source station.
- The `test_signals.csv` file contains strings for which you need to identify the source station.
- The `example.csv` file contains an example of a valid submission to the contest.

Thus, you need to predict the `source` column for the strings in the `test_signals.csv` file.

## What Needs to Be Done

You are required to upload a file named `answers.csv` to the system, in a format similar to that of `example.csv`, containing your predictions of the source station for the test transmissions. The evaluation metric is accuracy (the proportion of correct predictions). Your score for this task is calculated using the formula:

```
int(100 * max(5 * (accuracy - 0.8), 0)),
```

where accuracy is the proportion of correct predictions. Thus, to receive the maximum score, you need to achieve accuracy = 1. Note that naive features (characters at specific positions, individual character counts) are almost useless, because the cipher — a separate alphabet permutation for each string, plus a possible reversal — destroys the association with specific characters and the reading direction. Look for features that are invariant under these transformations.

## Answer

Language: (make) yandexdataschool. Submission by text or file upload; 100 attempts.
