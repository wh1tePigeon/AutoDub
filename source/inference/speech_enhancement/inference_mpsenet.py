import os
from re import S
import torch
import hydra
import sys
from pathlib import Path
import torchaudio as ta
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from source.model.mpsenet.generator import MPNet
from source.datasets.mpsenet.dataset import mag_pha_stft, mag_pha_istft
from source.utils.util import prepare_device
from source.utils.util import CONFIGS_PATH


CONFIG_MPSNET = CONFIGS_PATH / 'mpsnet'


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


@hydra.main(config_path=str(CONFIG_MPSNET), config_name="inf")
def main(cfg):
    inference_mpsenet(cfg)


if __name__ == '__main__':
    main()
