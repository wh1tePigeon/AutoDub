import os
from re import S
import torch
import hydra
import sys
from pathlib import Path
import torchaudio as ta
import torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from source.model.mpsenet.generator import MPNet
from source.datasets.mpsenet.dataset import mag_pha_stft, mag_pha_istft
from source.utils.util import prepare_device
from source.utils.util import CONFIGS_PATH


CONFIG_MPSENET = CONFIGS_PATH / 'mpsenet'


def inference_mpsenet(cfg):
    device, device_ids = prepare_device(cfg["n_gpu"])
    model = MPNet(cfg)
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    checkpoint = torch.load(cfg["checkpoint_path"], map_location=device)
    state_dict = checkpoint["generator"]
    model.load_state_dict(state_dict)
    model.eval()

    filepath = cfg["filepath"]
    assert os.path.exists(filepath)

    output_dir = cfg["output_dir"]
    sr = cfg["sampling_rate"]

    if os.path.isfile(filepath):
        audio, fs = ta.load(filepath)
        if fs != sr:
            audio = ta.functional.resample(audio, fs, sr)
            fs = sr

        # move audio to device
        audio = audio.to(device)

        with torch.inference_mode():
            # norm_factor = torch.sqrt(len(audio) / torch.sum(audio ** 2.0)).to(device)
            # audio = audio * norm_factor

            cfg_ft = {
                "n_fft" : cfg["n_fft"],
                "hop_size" : cfg["hop_size"],
                "win_size" : cfg["win_size"],
                "compress_factor" : cfg["compress_factor"]
            }

            noisy_amp, noisy_pha, noisy_com = mag_pha_stft(audio, **cfg_ft)
            amp_g, pha_g, com_g = model(noisy_amp, noisy_pha)
            audio_clean = mag_pha_istft(amp_g, pha_g, **cfg_ft)
            #speech = audio_clean / norm_factor
            # audio = audio / norm_factor

            speech = audio_clean
            background_len = audio.shape[-1]
            speech_len = speech.shape[-1]

            if background_len <= speech_len:
                speech = speech[:, :background_len]
            else:
                padding_size = background_len - speech_len
                speech = F.pad(speech, pad=(0, padding_size), mode='constant', value=0)

            background = audio - speech

            filename = filepath.split(".")[0].split("/")[-1]

            directory_save_file = os.path.join(output_dir, filename)
            os.makedirs(directory_save_file, exist_ok=True)
            
            speech_save_path = os.path.join(directory_save_file, (filename + "_speech.wav"))
            background_save_path = os.path.join(directory_save_file, (filename + "_background.wav"))
            audio_save_path = os.path.join(directory_save_file, (filename + "_orig.wav"))

            speech = speech.to("cpu")
            background = background.to("cpu")

            ta.save(speech_save_path, speech, sample_rate=sr)
            ta.save(background_save_path, background, sample_rate=sr)
            ta.save(audio_save_path, audio, sample_rate=sr)

            return [speech_save_path, background_save_path]


@hydra.main(config_path=str(CONFIG_MPSENET), config_name="inf")
def main(cfg):
    cfg["filepath"] = "/home/comp/Рабочий стол/RaM/segment_2_cutted.wav"
    inference_mpsenet(cfg)


if __name__ == '__main__':
    main()
