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
from source.utils.fader import OverlapAddFader

CONFIG_BSRNN_PATH = CONFIGS_PATH / 'bsrnn'
BSRNN_CHECKPOINT_PATH = CHECKPOINTS_DEFAULT_PATH / 'bsrnn' / 'main.pth'
BSRNN_OUTPUT_PATH = OUTPUT_DEFAULT_PATH / 'bsrnn'
INPUT_PATH = "/home/comp/Рабочий стол/AutoDub/input/1.wav"
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
        audio = audio.reshape(1, 1, -1)
  
        # move audio to gpu
        audio = audio.to(device)

        with torch.inference_mode():
            def forward(audio):
                _, output = model({"audio": {"mixture": audio},})
                return output["audio"]
            
            if audio.shape[-1] / REQUIRED_SR > 10:
                fader = OverlapAddFader(window_type="hann",
                                        chunk_size_second=6.0,
                                        hop_size_second=0.5,
                                        fs=44100,
                                        batch_size=6)
                
                output = fader(audio,
                                lambda a: forward(a))
                
            else:
                output = forward(audio)
            
            speech = output["audio"]["speech"]
            speech = speech.reshape(1, -1)
            audio = audio.reshape(1, -1)
            background = audio - speech

            filename = filepath.split(".")[0].split("/")[-1]

            directory_save_file = os.path.join(directory_save, filename)
            if not os.path.exists(directory_save_file):
                os.mkdir(directory_save_file)
            
            speech_save_path = os.path.join(directory_save_file, (filename + "_speech.wav"))
            background_save_path = os.path.join(directory_save_file, (filename + "_background.wav"))

            speech = speech.to("cpu")
            background = background.to("cpu")

            ta.save(speech_save_path, speech, sample_rate=REQUIRED_SR)
            ta.save(background_save_path, background, sample_rate=REQUIRED_SR)


if __name__ == "__main__":
    inference_bsrnn()