#!/usr/bin/env python3
"""Large-input timing and memory for task D.

Usage:  python3 bench.py [./solution] [N] > evidence_output.txt
Generates valid event sequences of length N (default 10^7), runs the solution on each,
reports wall time and peak memory, and checks a 20000-event prefix against the naive simulation.
"""
import os, random, resource, subprocess, sys, tempfile, time
from stress import brute, gen

exe = sys.argv[1] if len(sys.argv) > 1 else './solution'
N = int(sys.argv[2]) if len(sys.argv) > 2 else 10 ** 7

def adversarial(n):
    # half of the events are joins to queue 1, then both counters are closed/opened in turn:
    # every close moves a whole queue, every open moves half of a ~n/2-long queue
    s = 'a' * (n // 2 - 2) + '><' + '][' * (n // 4 - 10) + 'A' * 10
    return s[:n]

cases = [
    ('random, weights join/serve/close/open = 5/4/2/2', lambda: gen(N, random.Random(7), [5, 4, 2, 2])[0]),
    ('random, frequent close/open = 3/2/6/6', lambda: gen(N, random.Random(7), [3, 2, 6, 6])[0]),
    ('random, mostly joins = 10/1/1/1 (long queues)', lambda: gen(N, random.Random(7), [10, 1, 1, 1])[0]),
    ('adversarial: n/2 joins, then n/2 alternating close/open', lambda: adversarial(N)),
]
print(f'N = {N}')
for name, make in cases:
    s = make()
    with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False) as f:
        f.write(f'{len(s)}\n{s}\n'); path = f.name
    t = time.time()
    with open(path) as fin:
        out = subprocess.run([exe], stdin=fin, capture_output=True, text=True).stdout.strip()
    dt = time.time() - t
    rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024          # MiB on Linux, max over children so far
    pre = s[:20000]
    small = subprocess.run([exe], input=f'{len(pre)}\n{pre}\n', capture_output=True, text=True).stdout.strip()
    print(f'{name}: events {len(s)}, served {len(out)}, wall {dt:.2f}s, peak RSS so far {rss:.0f} MiB, 20000-event prefix equals naive simulation: {small == brute(pre)}', flush=True)
    os.unlink(path)
