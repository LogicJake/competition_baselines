#!/bin/bash

for i in `seq 0 9`; do
    python3 run_folds.py ${i}
done

python3 merge_folds.py
