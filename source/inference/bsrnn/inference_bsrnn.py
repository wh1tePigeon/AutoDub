import os
from pathlib import Path
import sys
import hydra
from hydra.utils import instantiate
import torch
import torchaudio as ta
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from source.utils.util import prepare_device, CONFIGS_PATH, CHECKPOINTS_DEFAULT_PATH, OUTPUT_DEFAULT_PATH
from source.utils.process_input_audio import load_n_process_audio

CONFIG_BSRNN_PATH = CONFIGS_PATH / 'bsrnn'
BSRNN_CHECKPOINT_PATH = CHECKPOINTS_DEFAULT_PATH / 'bsrnn' / 'main.pth'
BSRNN_OUTPUT_PATH = OUTPUT_DEFAULT_PATH / 'bsrnn'
INPUT_PATH = "/home/comp/Рабочий стол/AutoDub/input"
REQUIRED_SR = 44100


@hydra.main(config_path=str(CONFIG_BSRNN_PATH), config_name="main")
def inference_bsrnn(cfg):
    device, device_ids = prepare_device(cfg["n_gpu"])
    model = instantiate(cfg["arch"])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    checkpoint = torch.load(BSRNN_CHECKPOINT_PATH, map_location=device)
    state_dict = checkpoint["state_dict"]
    model.load_state_dict(state_dict)
    model.eval()


    filepath = INPUT_PATH
    directory_save = BSRNN_OUTPUT_PATH

    if os.path.isfile(filepath):
        audio, filepath = load_n_process_audio(filepath, directory_save, REQUIRED_SR)

        track_name = os.path.basename(filepath).split(".")[0]
        track = [track_name]
        
        # move audio to gpu
        audio = audio.to(device)

        with torch.inference_mode():
            # create batch
            batch = {
                    "audio": {
                            "mixture": audio,
                    },
                    "track": track,
            }

            # process batch
            output = model(batch)
            if type(output) is dict:
                batch.update(output)
            else:
                raise Exception("change type of model")
            print(output)
            # i run out of memory((



if __name__ == "__main__":
    inference_bsrnn()