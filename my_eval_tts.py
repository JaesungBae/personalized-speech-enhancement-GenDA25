import warnings
import re
import os
from tqdm import tqdm
import glob
import torch
from torch.nn.functional import cosine_similarity
import librosa
import numpy as np
import sys
from collections import defaultdict

# sys.path.append('../../UTMOS-demo')

# from score import Score
import torchaudio

def get_spks():
    spks = list(os.listdir('../data'))
    return sorted(spks)

def get_reference_speech(spk):
    reference_wav = glob.glob(os.path.join('../data', spk, 'reference_wav', '*.wav'))
    return reference_wav[0]


def calc_spk_metrics(ref_path, pred_paths, classifier):
    scores = []
    ref, _ = librosa.load(ref_path, sr=16000)
    ref = librosa.util.normalize(ref)
    ref = torch.tensor(ref).unsqueeze(0).cuda()
    preds = []
    for pred_path in pred_paths:
        pred, _ = librosa.load(pred_path, sr=16000)
        pred = librosa.util.normalize(pred)
        pred = torch.tensor(pred).unsqueeze(0).cuda()
        preds.append(pred)

    for i, pred in enumerate(preds):
        ref_embed = classifier.encode_batch(ref).squeeze(0)
        pred_embed = classifier.encode_batch(pred).squeeze(0)
        score = cosine_similarity(ref_embed, pred_embed, dim=1)
        scores.append(float(score))
    return scores


def cal_resemblyzer_spk_distance(ref_path, pred_paths, encoder):
    def load_wav(wav):
        return preprocess_wav(Path(wav))

    scores = []
    ref = load_wav(ref_path)
    # ref, _ = librosa.load(ref_path, sr=16000)
    # ref = librosa.util.normalize(ref)
    # ref = torch.tensor(ref).unsqueeze(0).cuda()
    preds = []
    for pred_path in pred_paths:
        pred = load_wav(pred_path)
        # pred, _ = librosa.load(pred_path, sr=16000)
        # pred = librosa.util.normalize(pred)
        # pred = torch.tensor(pred).unsqueeze(0).cuda()
        preds.append(pred)

    ref_embed = torch.tensor(encoder.embed_utterance(ref))
    for i, pred in enumerate(preds):
        pred_embed = torch.tensor(encoder.embed_utterance(pred))
        # ref_embed = classifier.encode_batch(ref).squeeze(0)
        # pred_embed = classifier.encode_batch(pred).squeeze(0)
        score = cosine_similarity(ref_embed, pred_embed, dim=0)
        scores.append(float(score))
    return scores

def cal_speechbrain_ecapa(ref_path, pred_paths, encoder):
    def load_wav(wav):
        return preprocess_wav(Path(wav))

    scores = []
    ref = load_wav(ref_path)
    # ref, _ = librosa.load(ref_path, sr=16000)
    # ref = librosa.util.normalize(ref)
    # ref = torch.tensor(ref).unsqueeze(0).cuda()
    preds = []
    for pred_path in pred_paths:
        pred = load_wav(pred_path)
        # pred, _ = librosa.load(pred_path, sr=16000)
        # pred = librosa.util.normalize(pred)
        # pred = torch.tensor(pred).unsqueeze(0).cuda()
        preds.append(pred)

    ref_embed = torch.tensor(encoder.embed_utterance(ref))
    for i, pred in enumerate(preds):
        pred_embed = torch.tensor(encoder.embed_utterance(pred))
        # ref_embed = classifier.encode_batch(ref).squeeze(0)
        # pred_embed = classifier.encode_batch(pred).squeeze(0)
        score = cosine_similarity(ref_embed, pred_embed, dim=0)
        scores.append(float(score))
    return scores
    


def calc_utmos(pred_paths):
    scores = []
    ckpt_path = "epoch=3-step=7459.ckpt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for pred_path in pred_paths:
        wav, sr = torchaudio.load(pred_path)
        assert sr == 16000
        wav = torchaudio.resample(wav, sr, 16000)
        scorer = Score(ckpt_path=ckpt_path, input_sample_rate=sr, device=device)
        score = scorer.score(wav.to(device))
        print(score)
        scores.append(score[0])
    return scores


class WER_cal(object):
    def __init__(self):
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
        from datasets import load_dataset
        from jiwer import wer

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        model_id = "openai/whisper-large-v3"

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        model.to(device)
        model.eval()

        processor = AutoProcessor.from_pretrained(model_id)

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device,
        )
        self.wer = wer
    
    def normalize_text(self, text):
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def cal_wer(self, pred_paths, texts):
        predicts, targets = '', ''
        for pred_path, target in tqdm(zip(pred_paths, texts), desc='Calculating WER'):
            predict = self.pipe(pred_path)['text']
            predict = self.normalize_text(predict)
            target = self.normalize_text(target)
            predicts = f'{predicts} {predict}'
            targets = f'{targets} {target}'
        wer_score = self.wer(targets, predicts)
        return wer_score



if __name__ == '__main__':
    # Ignore all warnings
    warnings.filterwarnings("ignore")
    # eval_metrics = ['SECS', 'Whipser']
    # eval_metrics = ['Resemblyzer']
    eval_metrics = ['SpeechBrain_Ecapa']
    spks = get_spks()
    lrs = [1e-6]
    
    models = ['gt', 'xttsv2', 'speecht5', 'yourtts']

    if 'SECS' in eval_metrics:
        from speechbrain.pretrained import EncoderClassifier
        classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb", savedir="pretrained_models/spkrec-xvect-voxceleb", run_opts={"device":"cuda"})
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
                tmp_utmos = calc_utmos(wav_list)
                # tmp_scores = calc_spk_metrics(referecne_wav_path, wav_list, classifier)
                # scores += tmp_scores
                utmos_scores += tmp_utmos
            assert len(scores) == 50 * len(spks)
            out[model] = {
                "SECS": np.mean(scores),
                "95CI": (np.percentile(scores, 97.5) - np.percentile(scores, 2.5)) / 2,
            }
            print(model)
            print('mean score:', np.mean(scores))
            # print('std nscore:', sum([(s - sum(scores) / len(scores)) ** 2 for s in scores]) / len(scores) ** 0.5)
            print('95% CI:', (np.percentile(scores, 97.5) - np.percentile(scores, 2.5)) / 2)
        with open('../eval_out/tts results.txt', 'w') as f:
            for model in out:
                tmp = f'{model}\t'
                for k, v in out[model].items():
                    tmp += f'{v:.3f}\t'
                f.write(tmp[:-1] + '\n')
    
    if 'Whipser' in eval_metrics:
        wer_cal = WER_cal()
        out = {}
        sentences = defaultdict(list)
        with open('/home/jb82/workspace_2024/GenDA_Challenge/Baseline/synth_sentences.csv', 'r') as f:
            lines = f.readlines()
            for line in lines:
                spk_id, sentence = line.strip().split('\t')
                spk_id = spk_id.split('/')[0]
                sentences[spk_id].append(sentence)

        # Use different setences set for gt
        gt_sentences = {}
        with open('../test_data.csv', 'r') as f:
            lines = f.readlines()
            for line in lines:
                w, t = line.strip().split('\t')
                gt_sentences[w] = t

        for model in models:
            total_score = 0
            for spk in tqdm(spks):
                if model == 'gt':
                    wav_list = glob.glob(os.path.join('../data', spk, 'wavs', '*.wav'))
                    assert len(wav_list) == 50
                    wav_list = sorted(wav_list)

                    tmp_sentences = []
                    for w in wav_list:
                        t = gt_sentences['/'.join(w.split('/')[-3:])]
                        tmp_sentences.append(t)
                    score = wer_cal.cal_wer(wav_list, tmp_sentences)
                else:
                    wav_list = glob.glob(os.path.join('../synth_data', spk, f'{spk}_{model}_*.wav'))
                    assert len(wav_list) == 50
                    wav_list = sorted(wav_list)
                    assert len(sentences[spk]) == 50    
                    score = wer_cal.cal_wer(wav_list, sentences[spk])

                total_score += score
                print(spk, 'wer', score)
            out[model] = {'WER': total_score / len(spks)}

        with open('../eval_out/tts_whisper_result.txt', 'w') as f:
            for model in out:
                tmp = f'{model}\t'
                for k, v in out[model].items():
                    tmp += f'{v:.6f}\t'
                f.write(tmp[:-1] + '\n')
    
    if 'Resemblyzer' in eval_metrics:
        from resemblyzer import VoiceEncoder, preprocess_wav
        from pathlib import Path
        import numpy as np

        encoder = VoiceEncoder()
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
                tmp = cal_resemblyzer_spk_distance(referecne_wav_path, wav_list, encoder)
                print(tmp)
                scores += tmp
            assert len(scores) == 50 * len(spks)
            out[model] = {
                "Resemblyzer": np.mean(scores),
                "95CI": (np.percentile(scores, 97.5) - np.percentile(scores, 2.5)) / 2,
            }
            print(model)
            print('mean score:', np.mean(scores))
            # print('std nscore:', sum([(s - sum(scores) / len(scores)) ** 2 for s in scores]) / len(scores) ** 0.5)
            print('95% CI:', (np.percentile(scores, 97.5) - np.percentile(scores, 2.5)) / 2)
        with open('../eval_out/tts_resemblyzer_results.txt', 'w') as f:
            for model in out:
                tmp = f'{model}\t'
                for k, v in out[model].items():
                    tmp += f'{v:.3f}\t'
                f.write(tmp[:-1] + '\n')
    
    if 'SpeechBrain_Ecapa' in eval_metrics:
        from speechbrain.pretrained import EncoderClassifier
        # classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb", savedir="pretrained_models/spkrec-xvect-voxceleb", run_opts={"device":"cuda"})
        classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
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
                tmp = calc_spk_metrics(referecne_wav_path, wav_list, classifier)
                scores += tmp
            assert len(scores) == 50 * len(spks)
            out[model] = {
                "SECS": np.mean(scores),
                "95CI": (np.percentile(scores, 97.5) - np.percentile(scores, 2.5)) / 2,
            }
            print(model)
            print('mean score:', np.mean(scores))
            # print('std nscore:', sum([(s - sum(scores) / len(scores)) ** 2 for s in scores]) / len(scores) ** 0.5)
            print('95% CI:', (np.percentile(scores, 97.5) - np.percentile(scores, 2.5)) / 2)
        with open('../eval_out/tts_speechbrain_ecapa_results.txt', 'w') as f:
            for model in out:
                tmp = f'{model}\t'
                for k, v in out[model].items():
                    tmp += f'{v:.3f}\t'
                f.write(tmp[:-1] + '\n')


