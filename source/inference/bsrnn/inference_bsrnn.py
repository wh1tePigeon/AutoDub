import os
from pathlib import Path
import sys
import hydra
from hydra.utils import instantiate
import torch
import torchaudio as ta
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from source.utils.util import prepare_device, CONFIGS_PATH, CHECKPOINTS_DEFAULT_PATH, OUTPUT_DEFAULT_PATH


CONFIG_BSRNN_PATH = CONFIGS_PATH / 'bsrnn'
BSRNN_CHECKPOINT_PATH = CHECKPOINTS_DEFAULT_PATH / 'bsrnn' / 'main.pth'
BSRNN_OUTPUT_PATH = OUTPUT_DEFAULT_PATH / 'bsrnn'
INPUT_PATH = "/home/comp/Рабочий стол/AutoDub/input"

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


    directory = INPUT_PATH
    directory_save = BSRNN_OUTPUT_PATH
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            directory_save_file = os.path.join(directory_save, filename)
            if not os.path.exists(directory_save_file):
                os.mkdir(directory_save_file)
            audio, fs = ta.load(filepath)

            # resample input
            if fs != cfg["arch"]["fs"]:
                audio = ta.functional.resample(
                    audio,
                    fs,
                    cfg["arch"]["fs"]
                )

            fs = cfg["arch"]["fs"]
            track_name = os.path.basename(filepath).split(".")[0]
            track = [track_name]
            #treat_batch_as_channels = False

            # process audio`s channels
            in_channel_audio = audio.shape[0]
            in_channel_model = cfg["arch"]["in_channel"]

            #if in_channel_audio != in_channel_model:
            #    if in_channel_audio == 1 and in_channel_model > 1:
            #        audio = audio.repeat(in_channel_model, 1)
            #    elif in_channel_audio > 1 and in_channel_model == 1:
            #        audio = audio[:, None, :]
            #        treat_batch_as_channels = True
            #        track = [track_name + f"_{i}" for i in range(in_channel_audio)]
            #else:
            #    raise ValueError(
            #        f"Cannot handle in_channel_audio={in_channel_audio} "
            #        f"and in_channel_model={in_channel_model}"
            #)

            if in_channel_audio == 1 and in_channel_model == 1:
                audio = audio[None, ...]

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