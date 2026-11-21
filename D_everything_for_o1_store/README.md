# D. "Everything for O(1)" store — analysis

Submit `solution.cpp` (the contest page selected C++23, GCC 14.1). The contest verdict was not reported in the source conversation.

## Build and run

```bash
make            # build ./solution
make test       # run the two sample cases from the statement
make stress     # compare 2,000 random attempts with a naive simulation (requires python3)
make bench      # large inputs, time, and memory -> evidence_output.txt
./solution < tests/sample1.in
```

## 1. What is required

There can be up to 10⁷ events, with a two-second time limit. A naive simulation using two queues is too slow: closing a checkout moves an entire queue, opening one moves half a queue, and alternating those events can produce quadratic time. We need O(1) time per event.

## 2. Understanding the rules

**Closing a checkout.** Customers from the closed queue move to the end of the other queue in reverse order. If checkout 1 closes:

```text
Q2_new = Q2 + reverse(Q1),   Q1_new = empty
```

**Opening a checkout.** Suppose the only open queue has m customers. A customer who is kth from the back is currently at position m − k + 1 and would take position k in the new queue. They move when k < m − k + 1, or equivalently k ≤ ⌊m/2⌋. Thus exactly the last ⌊m/2⌋ customers move, in reverse order.

Checks against the statement's samples:

- Sample 1, event 11: queue 2 is `2, 5, 6`; m = 3, so ⌊3/2⌋ = 1 customer moves — customer 6. This matches the statement's table.
- Sample 2: after `>`, queue 2 is `5, 4, 3, 2, 1`; m = 5, so two customers move. The new queue 1 is `1, 2`, while queue 2 remains `5, 4, 3`. The output is `12543`, as specified.

## 3. Key idea: one line instead of two queues

Write all customers as a single line L: queue 1 from front to back, followed by queue 2 **from back to front**.

```text
L = Q1[front .. back] | Q2[back .. front]
                        ^ boundary
```

Queue 1's front is the left end of L; queue 2's front is the right end. Their backs meet at the boundary. Each event then has a simple effect:

| Event | Effect on L |
|---|---|
| `a` | Insert a new customer to the left of the boundary |
| `b` | Insert to the right of the boundary |
| `A` | Remove from the left end |
| `B` | Remove from the right end |
| `>` | L is unchanged; move the boundary to the left end |
| `]` | L is unchanged; move the boundary to the right end |
| `<` | L is unchanged; put the boundary after ⌊m/2⌋ elements from the left |
| `[` | L is unchanged; put the boundary before ⌊m/2⌋ elements from the right |

**Why closing does not change L.** When checkout 1 closes, the new queue 2 is `Q2 + reverse(Q1)`. Queue 2 is written backward in L, so `reverse(Q2 + reverse(Q1)) = Q1 + reverse(Q2)`, exactly the old L. The only change is that the whole line now belongs to queue 2, placing the boundary at the left end. Closing checkout 2 is symmetric.

**Why opening does not change L.** Suppose checkout 1 opens. All of L is queue 2 in reverse order, so the back of queue 2 is at the left end. The last ⌊m/2⌋ customers in queue 2 are the first ⌊m/2⌋ elements of L, already ordered as the new queue 1 needs them. We only have to place the boundary after them. Opening checkout 2 is symmetric: the boundary goes at position m − ⌊m/2⌋.

## 4. Data structure

We need deletion at either end, insertion at a known position, and a move to the midpoint in O(1). An array-based doubly linked list with two pointers provides this.

- `nx[]` and `pv[]` hold the list. Nodes 0 and 1 are sentinels for the left and right ends; customer c is stored at node c + 1.
- `bnd` is the first node of the right part (or the right sentinel when that part is empty). `s1` is the size of the left part, also the index of `bnd` in the line.
- `mid` is the node at index ⌊m/2⌋ in the line; `mi` is its actual index.

**Maintaining the midpoint.** Each event changes the line length by at most one, so the desired index ⌊m/2⌋ moves by at most one; the actual index of `mid` also changes by at most one if an insertion or deletion occurs to its left. After each insertion or deletion, the code updates `mi` and moves `mid` to the desired position with a `while` loop that takes at most two steps. This gives worst-case O(1) time per event, with no amortization argument.

Boundary transitions:

- `>`: `bnd = nx[HEAD]`, `s1 = 0`;
- `]`: `bnd = TAIL`, `s1 = m`;
- `<`: `bnd = mid`, `s1 = ⌊m/2⌋`;
- `[`: `s1 = m − ⌊m/2⌋`; this index is ⌊m/2⌋ when m is even and ⌊m/2⌋ + 1 when m is odd, so `bnd = mid` or `bnd = nx[mid]`.

The code also handles deleting the node pointed to by `mid` (a line of one or two customers) and deleting the node pointed to by `bnd` from the right (when queue 2 had one customer).

## 5. Input, output, and memory

- Input is read in one `fread`; the event string is parsed character by character, skipping whitespace.
- The answer (last digits of served customer numbers) is accumulated in a buffer and written with one `fwrite`.
- Memory consists of two `int` arrays of n + 3 elements (80 MB at n = 10⁷), plus input and output buffers. Measured peak usage was 158 MiB under a 512 MB limit.

## 6. Verification

Run `make test` for the samples, `make stress` for random comparisons, and `make bench` for large inputs. The saved output of `make bench` is in `evidence_output.txt`.

- Both samples from the statement pass.
- `stress.py` generates valid random event sequences of length 2 to 1,000 with different event weights and compares the result against a naive Python-list simulation. Two runs of 3,000 attempts (seeds 1 and 2) checked 2,117 and 2,128 valid cases, with no mismatches. Attempts without any service event are skipped. To repeat those runs, use `python3 stress.py 1 3000` and `python3 stress.py 2 3000`.
- On inputs of 10⁷ events:

| Input | Time | Customers served |
|---|---|---|
| Random, join/serve/close/open weights 5/4/2/2 | 0.19 s | 3,637,548 |
| Random, frequent closing and opening, weights 3/2/6/6 | 0.20 s | 1,817,909 |
| Random, mostly joins, weights 10/1/1/1 (long queues) | 0.15 s | 833,200 |
| 5·10⁶ joins, then alternating closes and opens at both checkouts | 0.08 s | 10 |

The last input is a worst case for a naive solution: each event would move millions of customers.

The verification build used g++ 13.3 with `-O2 -std=c++20`. The code uses no C++23-specific features and should build with GCC 14.1 too, although it was not tested with that compiler version.

## Files

- `solution.cpp` — solution.
- `stress.py` — naive simulation, valid-sequence generator, and comparison.
- `bench.py` — large inputs, time, and memory.
- `tests/` — samples from the statement.
- `evidence_output.txt` — saved output of `bench.py`.
