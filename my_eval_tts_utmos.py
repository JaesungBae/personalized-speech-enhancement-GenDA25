import os
from tqdm import tqdm
import glob
import torch
import sys

sys.path.append('../../UTMOS-demo')

import numpy as np
import utmosv2

def get_spks():
    spks = list(os.listdir('../data'))
    return sorted(spks)

def get_reference_speech(spk):
    reference_wav = glob.glob(os.path.join('../data', spk, 'reference_wav', '*.wav'))
    return reference_wav[0]


def calc_utmos(pred_paths, model):
    scores = []

    for pred_path in pred_paths:
        mos = model.predict(input_path=pred_path)
        print(mos)
        scores.append(float(mos))
    return scores

if __name__ == '__main__':
    spks = get_spks()
    sizes = ['medium']
    lrs = [1e-6]
    
    models = ['gt', 'xttsv2', 'speecht5', 'yourtts']

    utmosv2_model = utmosv2.create_model(pretrained=True)
    utmosv2_model.cuda()
    utmosv2_model.eval()

    
    out = {}
    for model in models:
        scores = []
        utmos_scores = []
        for spk in tqdm(spks):
            referecne_wav_path = get_reference_speech(spk)

            if model == 'gt':
                wav_list = glob.glob(os.path.join('../data', spk, 'wavs', '*.wav'))
                assert len(wav_list) == 50
            else:
                wav_list = glob.glob(os.path.join('../synth_data', spk, f'{spk}_{model}_*.wav'))
                assert len(wav_list) == 50
            tmp_utmos = calc_utmos(wav_list, utmosv2_model)
            # tmp_scores = calc_spk_metrics(referecne_wav_path, wav_list, classifier)
            # scores += tmp_scores
            scores += tmp_utmos
        assert len(scores) == 50 * len(spks)
        out[model] = {
            "utmos": np.mean(scores),
            "95CI": (np.percentile(scores, 97.5) - np.percentile(scores, 2.5)) / 2,
        }
        print(model)
        print('mean score:', np.mean(scores))
        # print('std nscore:', sum([(s - sum(scores) / len(scores)) ** 2 for s in scores]) / len(scores) ** 0.5)
        print('95% CI:', (np.percentile(scores, 97.5) - np.percentile(scores, 2.5)) / 2)
    with open('../eval_out/tts_results_utmosv2.txt', 'w') as f:
        for model in out:
            tmp = f'{model}\t'
            for k, v in out[model].items():
                tmp += f'{v:.3f}\t'
            f.write(tmp[:-1] + '\n')

