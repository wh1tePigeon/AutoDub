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
from omegaconf import OmegaConf

#CONFIG_BSRNN_PATH = CONFIGS_PATH / 'bsrnn'
#BSRNN_CHECKPOINT_PATH = CHECKPOINTS_DEFAULT_PATH / 'bsrnn' / 'main.pth'
#BSRNN_OUTPUT_PATH = OUTPUT_DEFAULT_PATH / 'bsrnn'
#INPUT_PATH = "/home/comp/Рабочий стол/AutoDub/input/1.wav"
#REQUIRED_SR = 44100


#@hydra.main(config_path=str(CONFIG_BSRNN_PATH), config_name="main")
def inference_bsrnn(cfg):
    device, device_ids = prepare_device(cfg["n_gpu"])
    arch = OmegaConf.load(cfg["model"])
    #arch = OmegaConf.resolve(arch)
    model = instantiate(arch)
    #model = instantiate(cfg["model"])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    checkpoint = torch.load(cfg["checkpoint_path"], map_location=device)
    state_dict = checkpoint["state_dict"]
    model.load_state_dict(state_dict)
    model.eval()


    filepath = cfg["filepath"]
    output_dir = cfg["output_dir"]
    sr = cfg["sr"]

    if os.path.isfile(filepath):
        audio, filepath = load_n_process_audio(filepath, output_dir, sr)
        audio = audio.reshape(1, 1, -1)
  
        # move audio to gpu
        audio = audio.to(device)

        with torch.inference_mode():
            def forward(audio):
                _, output = model({"audio": {"mixture": audio},})
                return output["audio"]
            
            if audio.shape[-1] / sr > 10:
                fader = OverlapAddFader(window_type=cfg["window_type"],
                                        chunk_size_second=cfg["chunk_size_second"],
                                        hop_size_second=cfg["hop_size_second"],
                                        fs=sr,
                                        batch_size=cfg["batch_size"])
                
                output = fader(audio,
                                lambda a: forward(a))
                
            else:
                output = forward(audio)
            
            speech = output["audio"]["speech"]
            speech = speech.reshape(1, -1)
            audio = audio.reshape(1, -1)
            background = audio - speech

            filename = filepath.split(".")[0].split("/")[-1]

            directory_save_file = os.path.join(output_dir, filename)
            if not os.path.exists(directory_save_file):
                os.mkdir(directory_save_file)
            
            speech_save_path = os.path.join(directory_save_file, (filename + "_speech.wav"))
            background_save_path = os.path.join(directory_save_file, (filename + "_background.wav"))

            speech = speech.to("cpu")
            background = background.to("cpu")

            ta.save(speech_save_path, speech, sample_rate=sr)
            ta.save(background_save_path, background, sample_rate=sr)

            return [speech_save_path, background_save_path]


if __name__ == "__main__":
    inference_bsrnn()