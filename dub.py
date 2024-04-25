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
from source.inference import inference_bsrnn, inference_vad, inference_asr, translate_file_google


FILEPATH = "/home/comp/Рабочий стол/AutoDub/input/1.wav"

@hydra.main(config_path=str(CONFIGS_PATH), config_name="dub")
def dub(cfg):
    if FILEPATH != "":
        cfg["bsrnn"]["filepath"] = FILEPATH
    elif cfg["bsrnn"]["filepath"] is None:
        raise KeyError
    else:
        pass

    assert os.path.exists(FILEPATH)

    speech_path, background_path = inference_bsrnn(cfg["bsrnn"])

    cfg["vad"]["filepath"] = speech_path
    filepath, vad_boundaries_path = inference_vad(cfg["vad"])

    cfg["asr"]["filepath"] = filepath
    cfg["asr"]["boundaries"] = vad_boundaries_path
    transcribed_path = inference_asr(cfg["asr"])

    cfg["tr"]["filepath"] = transcribed_path
    translated_path = translate_file_google(cfg["tr"])

    print(translated_path)




if __name__ == "__main__":
    dub()