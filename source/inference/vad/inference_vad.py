import torchaudio as ta
import torch
from speechbrain.inference.VAD import VAD
import hydra
import os


#@hydra.main(config_path=str(CONFIG_VAD_PATH), config_name="main")
def inference_vad(cfg):
    path_to_input_dir = cfg["input dir"]
    path_to_output_dir = cfg["output dir"]

    # process all files in input directory
    for filename in os.listdir(path_to_input_dir):
        filepath = os.path.join(path_to_input_dir, filename)
        if os.path.isfile(filepath):
            directory_save_file = str(os.path.join(path_to_output_dir, filename))[:-4]

            if not os.path.exists(directory_save_file):
                os.mkdir(directory_save_file)

            audio, fs = ta.load(filepath)
            changed = False

            # resample to 16khz
            if fs != 16000:
                transofrm = ta.transforms.Resample(fs, 16000)
                audio = transofrm(audio)
                changed = True
            
            # make audio single channel
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)
                changed = True

            # save intermediate result
            if changed:
                filepath = os.path.join(directory_save_file, "changed.wav")
                ta.save(filepath, audio, 16000)

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

            path_to_log_file = os.path.join(directory_save_file, "vad_result.txt")
            vad.save_boundaries(boundaries, audio_file=filepath, save_path=path_to_log_file)


if __name__ == "__main__":
    cfg = {
        "input dir" : "/home/comp/Рабочий стол/AutoDub/input",
        "output dir" : "/home/comp/Рабочий стол/AutoDub/output/vad",
        "checkpoint dir" : "/home/comp/Рабочий стол/AutoDub/checkpoints/vad",
        "activation_th" : 0.8,
        "deactivation_th" : 0.0,
        "close_th" : 0.250,
        "len_th" : 0.250
    }
    inference_vad(cfg)