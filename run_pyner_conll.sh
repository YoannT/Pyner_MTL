#!/bin/bash

for ger_num in 1000 5000 10000 15000; do
    python main_pyner_conll.py \
        --ger_num ${ger_num}
done
