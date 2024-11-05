import json

import torch.utils.data as data
import torch
import torch.nn as nn
import os
import numpy as np
import torchaudio
import pandas as pd
import random
import soundfile as sf

np.random.seed(123)

PAD_INDEX=0
EPS = 1e-10
SAMPLING_RATE = 16000

class PSEData(data.Dataset):
    def __init__(self, csv_path, spk_id, mode='train'):
        data = pd.read_csv(csv_path)
        self.files = data[(data['spk']==spk_id) & (data['split']==mode)]['file'].values
        if mode=='train':
            noise_path = '/mnt/data3/musan/noise/free-sound'
            noise_files = os.listdir(noise_path)[60:]
        elif mode=='val':
            noise_path = '/mnt/data3/musan/noise/free-sound'
            noise_files = os.listdir(noise_path)[:60]
        else:
            noise_path = '/mnt/data3/musan/noise/sound-bible'
            noise_files = os.listdir(noise_path)

        self.noise_files = [os.path.join(noise_path, i) for i in noise_files if '.wav' in i]

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return (self.files[idx], self.noise_files)

def is_nan(t):
    inf = torch.isinf(t).any()
    nan = torch.isnan(t).any()
    if inf or nan:
        return True
    return False

def pad_noise(speech, noise, is_random_noise_cut=False):
    '''
    Cuts noise vector if speech vec is shorter
    Adds noise if speech vector is longer
    '''
    noise_len = noise.shape[1]
    speech_len = speech.shape[1]

    if speech_len > noise_len:
        repeat = (speech_len//noise_len) +1
        noise = torch.tile(noise, (1, repeat))
        diff = speech_len - noise.shape[1]
        noise = noise[:, :noise.shape[1]+diff]          
            
    elif speech_len < noise_len:
        if is_random_noise_cut:
            start_range = noise_len - speech_len
            start_idx = np.random.randint(0, start_range)
            noise = noise[:, start_idx:start_idx+speech_len]
        else:
            noise = noise[:,:speech_len]
    return noise

def mix_signals(speech, noise, desired_snr):    
    #calculate energies
    energy_s = torch.sum(speech**2, dim=-1, keepdim=True)
    energy_n = torch.sum(noise**2, dim=-1, keepdim=True)

    b = torch.sqrt((energy_s / energy_n) * (10 ** (-desired_snr / 10.)))
    return speech + b * noise

def collate_fn(data):
    orig_wav, fs = torchaudio.load(data[0][0])

    resample_fn = torchaudio.transforms.Resample(fs, 16000)
    mixtures = []
    targets = []

    for t in data:
        orig_wav, fs = torchaudio.load(t[0])
        orig_wav = resample_fn(orig_wav)
       
        musan_files = t[1]
        SNR = random.randrange(-5, 5)
        noise_file = random.choice(musan_files)
        mix_sig, fs = torchaudio.load(noise_file)
        mix_sig = pad_noise(orig_wav, mix_sig)
    
        mix_wav = mix_signals(orig_wav, mix_sig, SNR)
    
        mixtures.append(mix_wav)
        targets.append(orig_wav)
    
    max_len = max([i.shape[1] for i in mixtures])
    
    #print(f'before pad mixtures {[is_nan(i) for i in mixtures]}')
    mixtures = [torch.nn.functional.pad(i, (0, max_len-i.shape[1]), 'constant', PAD_INDEX).squeeze(0) for i in mixtures]
    
    targets = [torch.nn.functional.pad(i, (0, max_len-i.shape[1]), 'constant', PAD_INDEX).squeeze(0) for i in targets]
   
    mixtures = torch.stack(mixtures, dim=0)
    targets = torch.stack(targets, dim=0)
    #print(f'After stack {is_nan(mixtures)}')
    #print(f'After stack  targets {is_nan(targets)}')
    #print(f"x {is_nan(mixtures)}, y {is_nan(targets)}")
    return {"x":mixtures, "t":targets}



class Sampler:
    def __init__(self, csv_path, mode='train', is_random_noise_cut=False):
        self.is_random_noise_cut = is_random_noise_cut
        if mode=='train':
            noise_path = '/mnt/data3/musan/noise/free-sound'
            noise_files = os.listdir(noise_path)[60:]
        elif mode=='val':
            noise_path = '/mnt/data3/musan/noise/free-sound'
            noise_files = os.listdir(noise_path)[:60]
        elif mode=='test':
            noise_path = '/mnt/data3/musan/noise/sound-bible'
            noise_files = os.listdir(noise_path)  # 88
            print('number of noise files', len(noise_files))
        if 'ANNOTATIONS' in noise_files:
            noise_files.remove('ANNOTATIONS')
        if 'LICENSE' in noise_files:
            noise_files.remove('LICENSE')
        self.noise_path = noise_path
        self.noise_files = noise_files
        self.data = pd.read_csv(csv_path)
        
        self.curr_noise_idx = 0

    def _pad_source(self, source, total_len):
        repeat = (total_len//source.shape[1]) +1
        source = torch.tile(source, (1, repeat))
        source = source[:, :total_len] 
        return source
    
    def get_data_len(self, spk_id, mode):
        return len(self.data[(self.data['spk']==spk_id) & (self.data['split']==mode)])

    def sample_batch(self, spk_id, batch_size, mode='train', mix_snr=None):
        if mix_snr is not None:
            assert mode != 'train' and mode != 'val', "mix_snr is only for test mode"
        np.random.seed(42)

        data = self.data
        files = data[(data['spk']==spk_id) & (data['split']==mode)]

        #print(files[:3], len(files))
        sec = 4
        fs = 16000

        total_len = fs*sec

        mixtures = []
        targets = []

        while batch_size:
            idx = np.random.randint(0, len(files))
            source = files.iloc[idx]['file']
            source, fs = torchaudio.load(source)
            if fs != SAMPLING_RATE:
                source = torchaudio.functional.resample(source, fs, SAMPLING_RATE)
            if source.shape[1] < total_len:
                source = self._pad_source(source, total_len)
            elif source.shape[1] > total_len:
                start_range = source.shape[1] - SAMPLING_RATE * sec
                start_idx = np.random.randint(0, start_range)
                source = source[:,start_idx:start_idx+total_len]

            if mode != 'test':
                idx = np.random.randint(0, len(self.noise_files))
                noise, fs = torchaudio.load(os.path.join(self.noise_path, self.noise_files[idx]))
            else:
                # Fix noise in test mode
                print('curr_noise_idx', self.curr_noise_idx)
                noise, fs = torchaudio.load(os.path.join(self.noise_path, self.noise_files[self.curr_noise_idx]))
                self.curr_noise_idx += 1
            if fs != SAMPLING_RATE:
                noise= torchaudio.functional.resample(noise, fs, SAMPLING_RATE)
            noise = pad_noise(source, noise, self.is_random_noise_cut)
            # SNR = random.randrange(-5, 5) 

            if mix_snr is None:
                SNR = random.randrange(-5, 5)  # modify for jsbae
            else:
                SNR = mix_snr
            mixture = mix_signals(source, noise, SNR)
            assert mixture.shape[1]==total_len, f"Mixture dim does not match. Mixture {mixture.shape}, required size {total_len}" 
            mixtures.append(mixture)
            targets.append(source)
            batch_size-=1
        #print('--------------')
        #print(f"SPK {spk_id} {mixtures[0].sum()} {targets[0].sum()}")

        return {'x': torch.stack(mixtures, dim=1).squeeze(0), "t":torch.stack(targets, dim=1).squeeze(0)}


class SamplerAll:
    "Use all nosie files for "
    def __init__(self, csv_path, mode='train'):
        self.noises = []
        self._load_noise('/mnt/data3/musan/noise/free-sound')
        self._load_noise('/mnt/data3/musan/noise/sound-bible')
        print('# Total number of noises:', len(self.noises))

        self.data = pd.read_csv(csv_path)
    
    def _load_noise(self, noise_path):
        noise_files = os.listdir(noise_path)
        for noise_file in noise_files:
            if 'ANNOTATIONS' in noise_file:
                continue
            if 'LICENSE' in noise_file:
                continue
            noise, fs = torchaudio.load(os.path.join(noise_path, noise_file))
            if fs != SAMPLING_RATE:
                noise= torchaudio.functional.resample(noise, fs, SAMPLING_RATE)
            self.noises.append(noise)
        
    def _pad_source(self, source, total_len):
        repeat = (total_len//source.shape[1]) +1
        source = torch.tile(source, (1, repeat))
        source = source[:, :total_len] 
        return source
    
    def get_data_len(self, spk_id, mode):
        return len(self.data[(self.data['spk']==spk_id) & (self.data['split']==mode)])

    def sample_batch(self, spk_id, batch_size, mode='train', mix_snr=None):
        if mix_snr is not None:
            assert mode != 'train' and mode != 'val', "mix_snr is only for test mode"
        np.random.seed(42)

        data = self.data
        files = data[(data['spk']==spk_id) & (data['split']==mode)]

        #print(files[:3], len(files))
        sec = 4
        fs = 16000

        total_len = fs*sec

        mixtures = []
        targets = []

        noise_idx = 0
        while batch_size:
            idx = np.random.randint(0, len(files))
            source = files.iloc[idx]['file']
            source, fs = torchaudio.load(source)
            if fs != SAMPLING_RATE:
                source = torchaudio.functional.resample(source, fs, SAMPLING_RATE)
            if source.shape[1] < total_len:
                source = self._pad_source(source, total_len)
            elif source.shape[1] > total_len:
                start_range = source.shape[1] - SAMPLING_RATE * sec
                start_idx = np.random.randint(0, start_range)
                source = source[:,start_idx:start_idx+total_len]

            # choose random noise
            if mode == 'train':
                # only mix nosie when it is train mode
                noise_idx = np.random.randint(0, len(self.noises))
            else:
                noise_idx += 1  # it will start from 1, but ignore this.
            noise = pad_noise(source, self.noises[noise_idx])
            # SNR = random.randrange(-5, 5) 

            if mix_snr is None:
                SNR = random.randrange(-5, 5)  # modify for jsbae
            else:
                SNR = mix_snr
            mixture = mix_signals(source, noise, SNR)
            assert mixture.shape[1]==total_len, f"Mixture dim does not match. Mixture {mixture.shape}, required size {total_len}" 
            mixtures.append(mixture)
            targets.append(source)
            batch_size-=1
        #print('--------------')
        #print(f"SPK {spk_id} {mixtures[0].sum()} {targets[0].sum()}")

        return {'x': torch.stack(mixtures, dim=1).squeeze(0), "t":torch.stack(targets, dim=1).squeeze(0)}

class SamplerFixNoise:
    def __init__(self, csv_path, mode='train', is_random_noise_cut=False):
        self.is_random_noise_cut = is_random_noise_cut
        self.noise_path = '/mnt/data3/musan/noise/sound-bible'
        self.noise_dict = self._load_soundbible_noise()
        print('# Total number of noises:', len(self.noise_dict))

        self.data = pd.read_csv(csv_path).astype({'spk': 'str'})  # change speaker dtype to str
        with open('/home/jb82/workspace_2024/GenDA_Challenge/Baseline/spk_noise_set.json', 'r') as f:
            self.spk_noise_set = json.load(f)
    
    def _load_soundbible_noise(self):
        noise_dict = {}
        noise_files = os.listdir(self.noise_path)
        for noise_file in noise_files:
            if 'ANNOTATIONS' in noise_file:
                continue
            if 'LICENSE' in noise_file:
                continue
            noise, fs = torchaudio.load(os.path.join(self.noise_path, noise_file))
            if fs != SAMPLING_RATE:
                noise= torchaudio.functional.resample(noise, fs, SAMPLING_RATE)
            noise_dict[os.path.basename(noise_file)] = noise
        return noise_dict

    def _pad_source(self, source, total_len):
        repeat = (total_len//source.shape[1]) +1
        source = torch.tile(source, (1, repeat))
        source = source[:, :total_len] 
        return source
    
    def get_data_len(self, spk_id, mode):
        return len(self.data[(self.data['spk']==str(spk_id)) & (self.data['split']==mode)])

    def sample_batch(self, spk_id, batch_size, mode='train', mix_snr=None):
        if mix_snr is not None:
            assert mode != 'train' and mode != 'val', "mix_snr is only for test mode"
        np.random.seed(42)

        data = self.data
        files = data[(data['spk']==str(spk_id)) & (data['split']==mode)]
        noise_files = self.spk_noise_set[str(spk_id)]

        #print(files[:3], len(files))
        sec = 4
        fs = 16000

        total_len = fs*sec

        mixtures = []
        targets = []

        noise_cnt = [0] * len(noise_files)
        while batch_size:
            # Load sources
            idx = random.randint(0, len(files) - 1)
            source = files.iloc[idx]['file']
            source, fs = torchaudio.load(source)
            if fs != SAMPLING_RATE:
                source = torchaudio.functional.resample(source, fs, SAMPLING_RATE)
            if source.shape[1] < total_len:
                source = self._pad_source(source, total_len)
            elif source.shape[1] > total_len:
                start_range = source.shape[1] - SAMPLING_RATE * sec
                start_idx = random.randint(0, start_range - 1)
                source = source[:,start_idx:start_idx+total_len]

            # Load noise
            if mode != 'test':
                # Randomly choose noise
                idx = random.randint(0, len(noise_files) - 1)
                noise_cnt[idx] += 1
                noise, fs = torchaudio.load(os.path.join(self.noise_path, noise_files[idx]))
            else:
                # Fix noise in test mode
                print('curr_noise_idx', self.curr_noise_idx)
                noise, fs = torchaudio.load(os.path.join(self.noise_path, noise_files[self.curr_noise_idx]))
                self.curr_noise_idx += 1
            if fs != SAMPLING_RATE:
                noise= torchaudio.functional.resample(noise, fs, SAMPLING_RATE)
            noise = pad_noise(source, noise, self.is_random_noise_cut)
            # SNR = random.randrange(-5, 5) 

            # Mix noise
            if mix_snr is None:
                SNR = random.randrange(-5, 5)  # modify for jsbae
            else:
                SNR = mix_snr
            mixture = mix_signals(source, noise, SNR)
            assert mixture.shape[1]==total_len, f"Mixture dim does not match. Mixture {mixture.shape}, required size {total_len}" 
            mixtures.append(mixture)
            targets.append(source)
            batch_size-=1
        #print('--------------')
        #print(f"SPK {spk_id} {mixtures[0].sum()} {targets[0].sum()}")

        return {'x': torch.stack(mixtures, dim=1).squeeze(0), "t":torch.stack(targets, dim=1).squeeze(0)}

if __name__ == "__main__":
    test_speakers = [19, 26, 39, 40, 78, 83, 87, 89, 118, 125, 163, 196, 198, 200, 201, 250, 254, 307, 405, 446]
    split='val'
    sampler = Sampler('/home/anakuzne/utils/yourtts_60sec_set1.csv', split)
    #test_speakers = [1089, 121,  1284,  3575,  4446,  4970]
    #test_speakers = [0,  1,  2,  3,  4,  5,  6,  7,  8,  9]
    num_samples=10

    for spk in test_speakers:
        print(spk)
        save_path = f"/home/anakuzne/data/YourTTS_60sec_set1/{split}/spk_{spk}.pt"
        batch = sampler.sample_batch(spk, num_samples, split)
        #print(batch['x'].shape)
        torch.save(batch, save_path)
