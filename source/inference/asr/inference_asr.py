import os
from pathlib import Path
import sys
import hydra
from hydra.utils import instantiate
import torch
import torchaudio as ta
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from source.utils.util import prepare_device, CONFIGS_PATH, CHECKPOINTS_DEFAULT_PATH, OUTPUT_DEFAULT_PATH


#CONFIG_ASR_PATH = CONFIGS_PATH / 'asr'
#ASR_CHECKPOINT_PATH = CHECKPOINTS_DEFAULT_PATH / 'asr' / 'main.pth'
#ASR_OUTPUT_PATH = OUTPUT_DEFAULT_PATH / 'asr'
#INPUT_PATH = "/home/comp/Рабочий стол/AutoDub/input"

#@hydra.main(config_path=str(CONFIG_ASR_PATH), config_name="main")
def inference_asr(cfg):
    device, device_ids = prepare_device(cfg["n_gpu"])
    model = instantiate(cfg["arch"])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    checkpoint = torch.load(ASR_CHECKPOINT_PATH, map_location=device)
    state_dict = checkpoint["state_dict"]
    model.load_state_dict(state_dict)
    model.eval()


    filepath = INPUT_PATH
    directory_save = ASR_OUTPUT_PATH
    if os.path.isfile(filepath):
        filename = os.path.splitext(filepath)[0]
        directory_save_file = os.path.join(directory_save, filename)

        if not os.path.exists(directory_save_file):
            os.mkdir(directory_save_file)
        audio, fs = ta.load(filepath)

        resampled = False

        # resample input
        if fs != 16000:
            print("Wrong samplerate! Resample to 16 kHz")
            audio = ta.functional.resample(
                audio,
                fs,
                16000
            )
            resampled = True

        # save resampled audio
        if resampled:
            filepath = os.path.join(directory_save_file, (filename + "-resampled.wav"))
            ta.save(filepath, audio, 16000)
            print("Resampled audio saved in ", filepath)



        with torch.inference_mode():



if __name__ == "__main__":
    inference_asr()