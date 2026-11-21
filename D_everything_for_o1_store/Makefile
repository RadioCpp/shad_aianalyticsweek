CXX ?= g++
CXXFLAGS ?= -O2 -std=c++20 -Wall -Wextra

.PHONY: all test stress bench clean

all: solution

solution: solution.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

test: solution
	@for t in sample1 sample2; do \
		./solution < tests/$$t.in | diff -q - tests/$$t.out > /dev/null \
			&& echo "$$t: OK" || { echo "$$t: FAIL"; exit 1; }; \
	done

stress: solution
	python3 stress.py 1 2000

bench: solution
	python3 bench.py ./solution 10000000 | tee evidence_output.txt

clean:
	rm -f solution
