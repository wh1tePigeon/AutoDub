import os
from pathlib import Path

import hydra
from hydra.utils import instantiate
import torch
import torchaudio

from source.utils.util import prepare_device, CONFIGS_PATH, CHECKPOINTS_DEFAULT_PATH, OUTPUT_DEFAULT_PATH


CONFIG_BSRNN_PATH = CONFIGS_PATH / 'bsrnn'
BSRNN_CHECKPOINT_PATH = CHECKPOINTS_DEFAULT_PATH / 'bsrnn' / 'checkpoint.pth'
BSRNN_OUTPUT_PATH = OUTPUT_DEFAULT_PATH / 'bsrnn'

@hydra.main(config_path=str(CONFIG_BSRNN_PATH), config_name="main")
def inference_bsrnn(cfg, input_path, output_path=BSRNN_OUTPUT_PATH):
    device, device_ids = prepare_device(cfg["n_gpu"])
    model = instantiate(cfg["arch"])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    checkpoint = torch.load(BSRNN_CHECKPOINT_PATH, map_location=device)
    state_dict = checkpoint["state_dict"]

    model.load_state_dict(state_dict)

    model.eval()

    directory = input_path
    directory_save = output_path
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            directory_save_file = os.path.join(directory_save, filename)
            if not os.path.exists(directory_save_file):
                os.path.mkdir(directory_save_file)




            #target_audio, sample_rate = torchaudio.load(filepath)
            #target_mel = wav_to_mel(target_audio)
            #pred_audio = model.generator(target_mel).squeeze(0)
            #torchaudio.save(directory_save / filename, pred_audio, sample_rate)

if __name__ == "__main__":
    inference_bsrnn()