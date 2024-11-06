#!/bin/bash

#speakers=(0 1 2)[]
speakers=('VirtualF0' 'VirtualM0' 'VirtualF1' 'VirtualM1' 'VirtualF2' 'VirtualM2' 'VirtualF3' 'VirtualM3' 'VirtualF4' 'VirtualM4' 'VirtualF5' 'VirtualM5')
# sizes=('large' 'medium' 'small' 'tiny')
sizes=('medium' 'small' 'tiny')
#sizes=('tiny')
rates=(0.0001)
#rates=(0.00001)


for spk in ${speakers[@]}; do
    for s in ${sizes[@]}; do
        for r in ${rates[@]}; do
            echo "spk: $spk, size: $s, rate: $r"
            CUDA_VISIBLE_DEVICES=2 python my_run.py  -s $spk -r $r -i $s -p 'yourtts_synth_50utt_virtual' -m 'yourtts'
            CUDA_VISIBLE_DEVICES=2 python my_run.py  -s $spk -r $r -i $s -p 'yourtts_synth_250utt_virtual' -m 'yourtts'
        done
    done
done