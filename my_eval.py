import os
import pandas as pd
import torch
from tqdm import tqdm
from dataloaders import Sampler
import torchaudio
import exp_models as M
import soundfile as sf

from pystoi import stoi
from pesq import pesq
from exp_data import sdr_improvement, sdr
import numpy as np

from dataloaders import pad_noise, mix_signals

np.random.seed(123)

IS_SAVE_WAV = False
# PREFIX = '_-5to5_fix_optimload_randnoisecut'
PREFIX = '_-5to5_allnoise'
SNR_MAX = 5
# PREFIX = ''
# PREFIX = '_back_-5-25'
OUT_PATH = f'../eval_out{PREFIX}'
SAVE_PATH = f'/home/jb82/workspace_2024/GenDA_Challenge/Baseline/finetuned_checkpoints{PREFIX}'


def get_spks():
    spks = list(os.listdir('../data'))
    return sorted(spks)


def _pad_source(source, total_len):
    repeat = (total_len//source.shape[1]) +1
    source = torch.tile(source, (1, repeat))
    source = source[:, :total_len] 
    return source


def run_eval(size, model, lr, partition, spks, epoch, num_repeat=10):
    if not os.path.exists(OUT_PATH):
        os.makedirs(OUT_PATH, exist_ok=True)
    # for snr in [-2.5, 2.5, 7.5, 12.5, 17.5, 22.5]:
    snr_list = [-2.5, 2.5, 7.5, 12.5, 17.5, 22.5] 
    for snr in ['random']:
        if snr == 'random':
            snr = f'{snr}{SNR_MAX}'
        out = {}
        for spk_id in tqdm(spks):
            # Init paprameters.
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            if model != 'baseline':
                run_name = f"new_{model}_spk{spk_id}_{size}_{lr}_{partition}"
                if epoch != 'best':
                    checkpoint_path = os.path.join(SAVE_PATH, run_name, f'model_{epoch}.ckpt')
                else:
                    checkpoint_path = os.path.join(SAVE_PATH, run_name, 'model_best.ckpt')
                assert os.path.exists(checkpoint_path), checkpoint_path
                # csv_path = f'{SAVE_PATH}/csv_files/{partition}.csv'

            if model == 'baseline':
                run_name = f"baseline_spk{spk_id}_{size}"
                if size == 'medium':
                    checkpoint_path = '/home/jb82/workspace_2024/GenDA_Challenge/Baseline/baseline_checkpoints/Dec15_05-08-49_convtasnet_medium.pt'
                elif size == 'small':
                    checkpoint_path = '/home/jb82/workspace_2024/GenDA_Challenge/Baseline/baseline_checkpoints/Dec15_14-28-33_convtasnet_small.pt'
                else:
                    raise NotImplementedError
                # csv_path = f'/home/jb82/workspace_2024/GenDA_Challenge/Baseline/finetuned_checkpoints/csv_files/gt_50utt.csv'
            csv_path = f'{SAVE_PATH}/csv_files/gt_50utt.csv'
            
            spk_sdr = 0
            spk_isdr = 0
            spk_estoi = 0
            spk_pesq = 0
            # Init model.
            net, nparams, config = M.init_model(
                'convtasnet', size)

            # load weights from checkpoint
            net.load_state_dict(
                torch.load(checkpoint_path).get('model_state_dict'),
                strict=True)
            net.cuda()
            net.eval()

            # test_batch_sampler = Sampler(csv_path, 'test')
            # test_total_num = test_batch_sampler.get_data_len(int(spk_id), 'test')
            # print('# # of val samples:', test_total_num)
            
            # tmp_x, tmp_t = [], []
            # for _ in range(num_repeat):
            #     batch = test_batch_sampler.sample_batch(int(spk_id), test_total_num, 'test', mix_snr=snr)
            #     tmp_x.append(batch['x'].to(device))
            #     tmp_t.append(batch['t'].to(device))
            # x = torch.cat(tmp_x, dim=0)
            # t = torch.cat(tmp_t, dim=0)


            # Load test data manually
            x = []
            t = []
            noise_path = '/mnt/data3/musan/noise/sound-bible'
            noise_files = list(sorted(os.listdir(noise_path)))
            noise_files.remove('LICENSE')  # 88
            noise_files = noise_files * 2
            if SNR_MAX == 25:
                fixed_snr= [2.5, 7.5, 7.5, -2.5, 12.5, 7.5, 2.5, 2.5, 12.5, 12.5, 12.5, 2.5, 7.5, 22.5, 22.5, 2.5, -2.5, -2.5, -2.5, -2.5, 12.5, 17.5, 12.5, 17.5, -2.5, 22.5, 17.5, 7.5, 22.5, 22.5, 12.5, 2.5, 7.5, 2.5, -2.5, 17.5, 7.5, 2.5, 17.5, -2.5, 12.5, 22.5, 2.5, 17.5, 17.5, 7.5, 22.5, 2.5, 17.5, 22.5, 17.5, 17.5, 22.5, 17.5, 7.5, 7.5, 12.5, 7.5, -2.5, -2.5, -2.5, 12.5, 22.5, -2.5, 12.5, 12.5, 22.5, 17.5, 2.5, -2.5, 12.5, 17.5, 17.5, 22.5, 17.5, 2.5, 7.5, 17.5, -2.5, 7.5, 22.5, 12.5, 7.5, 12.5, 22.5, -2.5, 7.5, 7.5, 2.5, -2.5]
            elif SNR_MAX == 5:
                fixed_snr = [2.5, 2.5, 2.5, 0.0, 2.5, 2.5, 0.0, -2.5, 0.0, 0.0, 0.0, 2.5, 2.5, 0.0, -2.5, 0.0, 2.5, -2.5, 2.5, 2.5, 0.0, 0.0, 0.0, -2.5, 2.5, 2.5, -2.5, 0.0, -2.5, -2.5, -2.5, 2.5, 0.0, -2.5, 0.0, 2.5, 0.0, 0.0, 0.0, -2.5, 2.5, 0.0, 2.5, 2.5, 2.5, 0.0, -2.5, -2.5, -2.5, 2.5, 0.0, -2.5, -2.5, -2.5, 0.0, 0.0, -2.5, 0.0, -2.5, 0.0, -2.5, 0.0, 2.5, 0.0, 0.0, -2.5, 2.5, -2.5, -2.5, -2.5, 0.0, 2.5, 0.0, 0.0, 2.5, 2.5, 0.0, 2.5, 0.0, 2.5, 0.0, 0.0, -2.5, 0.0, 0.0, -2.5, 2.5, 2.5, 0.0, 0.0]
            else:
                raise NotImplementedError
            print(len(fixed_snr))
            curr_noise_idx = 0

            SAMPLING_RATE = 16000
            data = pd.read_csv(csv_path)
            for _ in range(num_repeat):
                files = data[(data['spk']==int(spk_id)) & (data['split']=='test')]
                for idx in range(len(files)):
                    # Load source
                    source = files.iloc[idx]['file']
                    source, fs = torchaudio.load(source)
                    if fs != SAMPLING_RATE:
                        source = torchaudio.functional.resample(source, fs, SAMPLING_RATE)
                    
                    # sec = 4
                    # total_len = SAMPLING_RATE * 4
                    # if source.shape[1] < total_len:
                    #     source = _pad_source(source, total_len)
                    # elif source.shape[1] > total_len:
                    #     start_range = source.shape[1] - SAMPLING_RATE * sec
                    #     start_idx = np.random.randint(0, start_range)
                    #     source = source[:,start_idx:start_idx+total_len]

                    # Load noise
                    noise, fs = torchaudio.load(os.path.join(noise_path, noise_files[curr_noise_idx]))
                    if fs != SAMPLING_RATE:
                        noise= torchaudio.functional.resample(noise, fs, SAMPLING_RATE)
                    noise = pad_noise(source, noise)

                    # Mix
                    if 'random' in snr:
                        _snr = fixed_snr[curr_noise_idx]
                    else:
                        _snr = snr
                    mixture = mix_signals(source, noise, _snr)
                
                    x.append(mixture.to(device).squeeze(0))
                    t.append(source.to(device).squeeze(0))
                    
                    curr_noise_idx += 1

            print('# number of sampels after repeat:', len(x))
            for i in range(len(x)):
                y_mini = M.make_2d(net(x[i]))

                # print(y_mini.size(), x[i].size(), t[i].size())
                # [1, 64000], [64000], [64000]
                
                if IS_SAVE_WAV:
                    y_out = y_mini.detach().cpu().numpy().reshape(-1, 1)
                    x_out = x[i].detach().cpu().numpy().reshape(-1, 1)
                    t_out = t[i].detach().cpu().numpy().reshape(-1, 1)
                    out_p = f"{OUT_PATH}/{spk_id}/"
                    if not os.path.exists(out_p):
                        os.makedirs(out_p)
                    out_fp = f"{out_p}/{spk_id}-{size}-{partition}-{i}-enhanced.wav"
                    out_xp = f"{out_p}/{spk_id}-{size}-{partition}-{i}-noisy.wav"
                    out_tp = f"{out_p}/{spk_id}-{size}-{partition}-{i}-clean.wav"
                    fs = 16000
                    sf.write(out_fp, y_out, fs)
                    sf.write(out_xp, x_out, fs)
                    sf.write(out_tp, t_out, fs)
                
                y_mini = y_mini.squeeze(0).detach().cpu().numpy()
                x_mini = x[i].detach().cpu().numpy()    
                t_mini = t[i].detach().cpu().numpy()    

                _sdr = sdr(y_mini, t_mini)
                _isdr = sdr_improvement(y_mini, t_mini, x_mini, 'mean')
                _estoi = stoi(t_mini, y_mini, fs, extended=True)
                _pesq = pesq(fs, t_mini, y_mini, 'wb')
                spk_sdr += _sdr
                spk_isdr += _isdr
                spk_estoi += _estoi
                spk_pesq += _pesq
            spk_sdr /= len(x)
            spk_isdr /= len(x)
            spk_estoi /= len(x)
            spk_pesq /= len(x)
            print('# Run:', run_name)
            print('# Prefix:', PREFIX, 'SNR_MAX:', SNR_MAX)
            print('# spk_isdr', spk_isdr, 'spk_sdr', spk_sdr, 'spk_estoi', spk_estoi, 'spk_pesq', spk_pesq)
            out[spk_id] = {'sdr': spk_sdr, 'isdr': spk_isdr, 'estoi': spk_estoi, 'pesq': spk_pesq}
        out['avg'] = {
            'sdr': sum([v['sdr'] for v in out.values()]) / len(out),
            'isdr': sum([v['isdr'] for v in out.values()]) / len(out),
            'estoi': sum([v['estoi'] for v in out.values()]) / len(out),
            'pesq': sum([v['pesq'] for v in out.values()]) / len(out),
        }
        print(f'eval_result_{size}_{model}_{lr}_{partition}_{snr}_e{epoch} DONE!')
        with open(f'{OUT_PATH}/eval_result_{size}_{model}_{lr}_{partition}_{snr}_e{epoch}.txt', 'w') as f:
            f.write('spk\tisdr\tsdr\testoi\tpesq\n')
            for spk_id, res in out.items():
                tmp = f'{spk_id}\t'
                for k in ['isdr', 'sdr', 'estoi', 'pesq']:
                    v = res[k]
                    tmp += f'{float(v):.3f}\t'
                f.write(tmp[:-1] + '\n')
    return out


if __name__ == '__main__':
    # Fix seed
    np.random.seed(123)
    torch.manual_seed(123)

    spks = get_spks()
    sizes = ['medium', 'small']
    # sizes = ['medium']
    # sizes = ['medium']
    lrs = [1e-6]
    num_repeat = 5
    # epochs = [101, 201]  # ['best']
    epochs = ['best']


    # models += ['gt']
    # partitions += ['gt_50utt']
    # models = ['baseline', 'yourtts', 'xttsv2', 'speecht5']
    # partitions = ['gt_50utt', 'yourtts_synth_50utt', 'xttsv2_synth_50utt', 'speecht5_synth_50utt']
    # models += ['yourtts', 'xttsv2', 'speecht5']
    # partitions += ['yourtts_synth_250utt', 'xttsv2_synth_250utt', 'speecht5_synth_250utt']
    
    # models += ['yourtts', 'yourtts']
    # partitions += ['yourtts_synth_50utt', 'yourtts_synth_250utt']

    models = []
    models += [('baseline', 'gt_50utt')]
    models += [('gt', 'gt_50utt')]
    models += [('speecht5', 'speecht5_synth_50utt'),
               ('speecht5', 'speecht5_synth_250utt')]
    models += [('yourtts', 'yourtts_synth_50utt'),
               ('yourtts', 'yourtts_synth_250utt')]
    models += [('xttsv2', 'xttsv2_synth_50utt'),
               ('xttsv2', 'xttsv2_synth_250utt')]



    print(spks)
    for size in sizes: 
        for model, partition in models:
            for lr in lrs:
                for epoch in epochs:
                    run_eval(size, model, lr, partition, spks, num_repeat=num_repeat, epoch=epoch)