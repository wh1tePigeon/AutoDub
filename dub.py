import os
import sys
import hydra
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from source.utils.util import CONFIGS_PATH
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

    print("Separating speech:")
    speech_path, background_path = inference_bsrnn(cfg["bsrnn"])

    print("Detecting voice activity:")
    cfg["vad"]["filepath"] = speech_path
    filepath, vad_boundaries_path = inference_vad(cfg["vad"])

    print("Speech recognition:")
    cfg["asr"]["filepath"] = filepath
    cfg["asr"]["boundaries"] = vad_boundaries_path
    transcribed_path = inference_asr(cfg["asr"])

    print("Translating")
    cfg["tr"]["filepath"] = transcribed_path
    translated_path = translate_file_google(cfg["tr"])


if __name__ == "__main__":
    dub()