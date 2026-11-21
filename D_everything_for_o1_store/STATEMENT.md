# D. "Everything for O(1)" store

> Transcribed from the contest page. The original screenshot is `statement.png`.

| | |
|---|---|
| Time limit | 2 seconds |
| Memory limit | 512 Mb |
| Input | standard input or input.txt |
| Output | standard output or output.txt |

The "Everything for O(1)" store has two checkout counters and many customers. Your task is to simulate the queues at these counters based on the recorded history of the store's operation.

You are given a chronological sequence of events encoded using the following characters:

- **a** — a new customer joins the end of the queue at the first checkout counter;
- **b** — a new customer joins the end of the queue at the second checkout counter;
- **A** — the first customer in the queue at the first checkout counter was served;
- **B** — the first customer in the queue at the second checkout counter was served;
- **>** — the first checkout counter closes;
- **]** — the second checkout counter closes;
- **<** — the first checkout counter opens;
- **[** — the second checkout counter opens.

When a checkout counter closes, all customers in its queue move to the end of the other queue in reverse order, starting with the last customer. That is, the person who was last in the queue moves first, followed by the person who was second to last, and so on. As a result, the person who was first in the queue at the counter that has just closed will be the last in the resulting queue.

When a closed checkout counter opens, customers from the queue at the other counter move to it, starting with the last customer, provided that their position in the new queue would be strictly less than their current position. The person who was last in the queue becomes the first in the new queue, the person who was second to last becomes the second, and so on.

The sequence of events is valid, meaning that:

- Only closed checkout counters can open;
- Only open checkout counters can close;
- Customers do not join queues at closed checkout counters;
- Closed checkout counters do not attempt to serve customers;
- Checkout counters do not serve customers if their queues are empty;
- At least one checkout counter is open at all times.

Customers are numbered starting from one, in the order in which they appear in the sequence of events. Initially, both checkout counters are open and both queues are empty.

## Input format

The first line of the input contains a natural number n (2 <= n <= 10^7) — the number of events.

The second line contains n characters describing the events according to the notation given above.

It is guaranteed that the input contains at least one query to process a request.

## Output format

In a single line, output the last digit of the customer number served for each service record. Output the answers in the order the service requests are processed, without using any separators.

## Scoring

The tests for this problem are divided into three groups. Points for each group are awarded only if all tests in that group and all tests in the preceding groups are passed.

- The group where n ≤ 1000 is worth 26 points.
- The group where n ≤ 200000 is worth 37 points.
- The group where n ≤ 10000000 is worth 37 points.

## Sample 1

Input

```
15
aaabA>bBBb<BBAa
```

Output

```
143256
```

## Sample 2

Input

```
12
aaaaa><AABBB
```

Output

```
12543
```

## Notes

Explanation of the first example:

| No. | Command | Explanation | Queue 1 | Queue 2 |
|---|---|---|---|---|
| 1 | a | Customer 1 joins Queue 1 | 1 | -- |
| 2 | a | Customer 2 joins Queue 1 | 1, 2 | -- |
| 3 | a | Customer 3 joins Queue 1 | 1, 2, 3 | -- |
| 4 | b | Customer 4 joins Queue 2 | 1, 2, 3 | 4 |
| 5 | A | Customer 1 is served from Queue 1 | 2, 3 | 4 |
| 6 | > | Checkout counter 1 closes | -- | 4, 3, 2 |
| 7 | b | Customer 5 joins Queue 2 | -- | 4, 3, 2, 5 |
| 8 | B | Customer 4 is served from Queue 2 | -- | 3, 2, 5 |
| 9 | B | Customer 3 is served from Queue 2 | -- | 2, 5 |
| 10 | b | Customer 6 joins Queue 2 | -- | 2, 5, 6 |
| 11 | < | Checkout counter 1 opens | 6 | 2, 5 |
| 12 | B | Customer 2 is served from Queue 2 | 6 | 5 |
| 13 | B | Customer 5 is served from Queue 2 | 6 | -- |
| 14 | A | Customer 6 is served from Queue 1 | -- | -- |
| 15 | a | Customer 7 joins Queue 1 | 7 | -- |

## Answer

Language selected on the page: C++23 (GCC 14.1). Submission by text or file upload.
