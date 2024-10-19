#!/bin/bash

#speakers=(0 1 2)[]
speakers=(237)
# sizes=('large' 'medium' 'small' 'tiny')
sizes=('medium')
#sizes=('tiny')
rates=(0.000001)
#rates=(0.00001)


for spk in ${speakers[@]}; do
    for s in ${sizes[@]}; do
        for r in ${rates[@]}; do
            echo "spk: $spk, size: $s, rate: $r"
            CUDA_VISIBLE_DEVICES=7 python my_run.py  -s $spk -r $r -i $s -p 'xttsv2_synth_40utt'
        done
    done
done