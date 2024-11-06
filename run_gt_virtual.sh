#!/bin/bash

#speakers=(0 1 2)[]
speakers=('VirtualF0' 'VirtualF1' 'VirtualF2' 'VirtualF3' 'VirtualF4' 'VirtualF5' 'VirtualM0' 'VirtualM1' 'VirtualM2' 'VirtualM3' 'VirtualM4' 'VirtualM5')
# sizes=('large' 'medium' 'small' 'tiny')
sizes=('medium' 'small' 'tiny')
# sizes=('tiny')
#sizes=('tiny')
rates=(0.0001)
#rates=(0.00001)


for spk in ${speakers[@]}; do
    for s in ${sizes[@]}; do
        for r in ${rates[@]}; do
            echo "spk: $spk, size: $s, rate: $r"
            CUDA_VISIBLE_DEVICES=5 python my_run.py  -s $spk -r $r -i $s -p 'gt_50utt_virtual' -m 'gt'
        done
    done
done