import glob
import os
import argparse
import json
from re import S
import torch
import librosa
import soundfile as sf
import sys
from pathlib import Path
import torchaudio as ta
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from source.model.mpsenet.generator import MPNet
from source.datasets.mpsenet.dataset import mag_pha_stft, mag_pha_istft
from source.utils.util import prepare_device
from hydra.utils import instantiate


def inference_mpsenet(cfg):
    device, device_ids = prepare_device(cfg["n_gpu"])
    model = MPNet(cfg)
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    checkpoint = torch.load(cfg["checkpoint_path"], map_location=device)
    state_dict = checkpoint["state_dict"]
    model.load_state_dict(state_dict)
    model.eval()

    filepath = cfg["filepath"]
    output_dir = cfg["output_dir"]
    sr = cfg["sampling_rate"]

    if os.path.isfile(filepath):
       # audio, sr = librosa.load(filepath, sr)
        audio, fs = ta.load(filepath)
        if fs != sr:
            audio = ta.functional.resample(audio, fs, sr)
            fs = sr

        # move audio to device
        audio = audio.to(device)

        with torch.inference_mode():
            norm_factor = torch.sqrt(len(audio) / torch.sum(audio ** 2.0)).to(device)
            audio = (audio * norm_factor).unsqueeze(0)
            noisy_amp, noisy_pha, noisy_com = mag_pha_stft(audio, **cfg["stft"])
            amp_g, pha_g, com_g = model(noisy_amp, noisy_pha)
            audio_clean = mag_pha_istft(amp_g, pha_g, **cfg["stft"])
            speech = audio_clean / norm_factor

            background = audio - speech

            filename = filepath.split(".")[0].split("/")[-1]

            directory_save_file = os.path.join(output_dir, filename)
            os.makedirs(directory_save_file, exist_ok=True)
            
            speech_save_path = os.path.join(directory_save_file, (filename + "_speech.wav"))
            background_save_path = os.path.join(directory_save_file, (filename + "_background.wav"))

            speech = speech.to("cpu")
            background = background.to("cpu")

            ta.save(speech_save_path, speech, sample_rate=sr)
            ta.save(background_save_path, background, sample_rate=sr)

            return [speech_save_path, background_save_path]



def main():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_clean_wavs_dir', default='VoiceBank+DEMAND/wavs_clean')
    parser.add_argument('--input_noisy_wavs_dir', default='VoiceBank+DEMAND/wav_noisy')
    parser.add_argument('--input_test_file', default='VoiceBank+DEMAND/test.txt')
    parser.add_argument('--output_dir', default='generated_files')
    parser.add_argument('--checkpoint_file', required=True)
    a = parser.parse_args()

    config_file = os.path.join(os.path.split(a.checkpoint_file)[0], 'config.json')
    with open(config_file) as f:
        data = f.read()

    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)

    torch.manual_seed(h.seed)
    global device
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    inference(a)


if __name__ == '__main__':
    main()
