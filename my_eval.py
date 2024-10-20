import os
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


OUT_PATH = '../eval_out'

def get_spks():
    spks = list(os.listdir('../data'))
    return sorted(spks)


def run_eval(size, model, lr, partition, spks, num_repeat=10):
    out = {}
    for spk_id in tqdm(spks):
        # Init paprameters.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        save_path = '/home/jb82/workspace_2024/GenDA_Challenge/Baseline/finetuned_checkpoints'
        if model != 'baseline':
            run_name = f"new_{model}_spk{spk_id}_{size}_{lr}_{partition}"
            checkpoint_path = os.path.join(save_path, run_name, 'model_best.ckpt')
            assert os.path.exists(checkpoint_path), checkpoint_path
            csv_path = f'/home/jb82/workspace_2024/GenDA_Challenge/Baseline/finetuned_checkpoints/csv_files/{partition}.csv'

        if model == 'baseline':
            if size == 'medium':
                checkpoint_path = '/home/jb82/workspace_2024/GenDA_Challenge/Baseline/baseline_checkpoints/Dec15_05-08-49_convtasnet_medium.pt'
            else:
                raise NotImplementedError
            csv_path = f'/home/jb82/workspace_2024/GenDA_Challenge/Baseline/finetuned_checkpoints/csv_files/gt_50utt.csv'
        
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

        test_batch_sampler = Sampler(csv_path, 'test')
        test_total_num = test_batch_sampler.get_data_len(int(spk_id), 'test')
        print('# # of val samples:', test_total_num)
        
        tmp_x, tmp_t = [], []
        for _ in range(num_repeat):
            batch = test_batch_sampler.sample_batch(int(spk_id), test_total_num, 'test')
            tmp_x.append(batch['x'].to(device))
            tmp_t.append(batch['t'].to(device))
        x = torch.cat(tmp_x, dim=0)
        t = torch.cat(tmp_t, dim=0)

        print('# number of sampels after repeat:', x.size(0))
        for i in range(x.size(0)):
            y_mini = M.make_2d(net(x[i]))

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
        spk_sdr /= x.size(0)
        spk_isdr /= x.size(0)
        spk_estoi /= x.size(0)
        spk_pesq /= x.size(0)
        out[spk_id] = {'sdr': spk_sdr, 'isdr': spk_isdr, 'estoi': spk_estoi, 'pesq': spk_pesq}
    out['avg'] = {
        'sdr': sum([v['sdr'] for v in out.values()]) / len(out),
        'isdr': sum([v['isdr'] for v in out.values()]) / len(out),
        'estoi': sum([v['estoi'] for v in out.values()]) / len(out),
        'pesq': sum([v['pesq'] for v in out.values()]) / len(out),
    }
    print(f'eval_result_{size}_{model}_{lr}_{partition} DONE!')
    with open(f'{OUT_PATH}/eval_result_{size}_{model}_{lr}_{partition}.txt', 'w') as f:
        f.write('spk\tisdr\tsdr\testoi\tpesq\n')
        for spk_id, res in out.items():
            tmp = f'{spk_id}\t'
            for k, v in res.items():
                tmp += f'{float(v):.3f}\t'
            f.write(tmp[:-1] + '\n')
    return out




if __name__ == '__main__':
    # Fix seed
    np.random.seed(123)
    torch.manual_seed(123)

    spks = get_spks()
    sizes = ['medium']
    lrs = [1e-6]

    models = ['gt']
    partitions = ['gt_50utt']

    models = ['baseline', 'gt', 'xttsv2', 'speecht5']
    partitions = ['baseline', 'gt_50utt', 'xttsv2_synth_50utt', 'speecht5_synth_50utt']

    print(spks)
    for size in sizes: 
        for model, partition in zip(models, partitions):
            for lr in lrs:
                run_eval(size, model, lr, partition, spks, num_repeat=10)