import torchaudio as ta
import torch
from speechbrain.inference.VAD import VAD
import hydra
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from source.utils.process_input_audio import load_n_process_audio
REQUIRED_SR = 16000

#@hydra.main(config_path=str(CONFIG_VAD_PATH), config_name="main")
def inference_vad(cfg):
    filepath = cfg["input_path"]
    directory_save = cfg["output_dir"]

    _, filepath = load_n_process_audio(filepath, directory_save, REQUIRED_SR)
    filename = filepath.split(".")[0].split("/")[-1]
    directory_save_file = os.path.join(directory_save, filename)
    if not os.path.exists(directory_save_file):
            os.mkdir(directory_save_file)

    # apply vad
    vad = VAD.from_hparams(source="speechbrain/vad-crdnn-libriparty",
                            savedir=cfg["checkpoint dir"])

    boundaries = vad.get_speech_segments(audio_file=filepath, apply_energy_VAD=False)
    #                                     apply_energy_VAD=True,
    #                                     activation_th=cfg["activation_th"],
    #                                     deactivation_th=cfg["deactivation_th"],
    #                                     close_th=cfg["close_th"],
    #                                     len_th=cfg["len_th"])

    # somehow this works better ¯\_(ツ)_/¯
    boundaries = vad.energy_VAD(filepath, boundaries,
                                activation_th=cfg["activation_th"],
                                deactivation_th=cfg["deactivation_th"])
    boundaries = vad.merge_close_segments(boundaries, close_th=cfg["close_th"])
    boundaries = vad.remove_short_segments(boundaries, len_th=cfg["len_th"])

    path_to_log_file = os.path.join(directory_save_file, (filename + "_boundaries.txt"))
    vad.save_boundaries(boundaries, audio_file=filepath, save_path=path_to_log_file)

    result = [filepath, path_to_log_file]

    return result

if __name__ == "__main__":
    cfg = {
        "input_path" : "/home/comp/Рабочий стол/AutoDub/input/speech.wav",
        "output_dir" : "/home/comp/Рабочий стол/AutoDub/output/vad",
        "checkpoint dir" : "/home/comp/Рабочий стол/AutoDub/checkpoints/vad",
        "activation_th" : 0.8,
        "deactivation_th" : 0.0,
        "close_th" : 0.250,
        "len_th" : 0.250
    }
    inference_vad(cfg)