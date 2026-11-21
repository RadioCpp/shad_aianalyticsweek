PY ?= python3

.PHONY: a b c d all evidence clean

all: a c d

a:
	cd A_mirror_for_algernon && $(PY) solve.py

b:
	cd B_inheritance && $(PY) solution.py data/train.csv data/test.csv submissions/

c:
	cd C_radio_interception && $(PY) solve_c.py data/train_signals.csv data/test_signals.csv answers.csv

d:
	$(MAKE) -C D_everything_for_o1_store test

# regenerate empirical results used in the write-ups
evidence:
	cd B_inheritance && $(PY) evidence.py data/train.csv data/test.csv data/starter.py > evidence_output.txt
	cd C_radio_interception && $(PY) evidence.py data/train_signals.csv data/test_signals.csv answers.csv probes > evidence_output.txt
	$(MAKE) -C D_everything_for_o1_store bench

clean:
	$(MAKE) -C D_everything_for_o1_store clean
