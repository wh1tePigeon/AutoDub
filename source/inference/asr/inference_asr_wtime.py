import os
import csv
import sys
import torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from source.utils.util import prepare_device
from source.utils.process_audio import load_n_process_audio
import whisper
import pandas as pd


# write output file
def create_output_file(result, output_file_path):
    df = pd.DataFrame(result)[['id', 'start', 'end', 'text']]

    #df = df.drop('seek', axis=1)
    df.to_csv(output_file_path, sep=';', index=False)


def inference_asr_wtime(cfg):
    m = cfg["model"]
    if m not in ["tiny", "base", "small", "medium", "large", "large-v1", "large-v2", "large-v3",
                    "tiny.en", "base.en", "small.en", "medium.en"]:
        raise KeyError
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model(name=m, download_root=cfg["checkpoint_path"]).to(device)
    model.transcribe

    filepath = cfg["filepath"]
    output_dir = cfg["output_dir"]
    sr = cfg["sr"]

    if os.path.isfile(filepath):
        audio, filepath = load_n_process_audio(filepath, output_dir, sr)

        filename = filepath.split(".")[0].split("/")[-1]
        directory_save_file = os.path.join(output_dir, filename)

        os.makedirs(directory_save_file, exist_ok=True)
                
        def transcribe_audio_whisper(filepath: str):
            result = model.transcribe(audio=filepath, word_timestamps=True)["segments"]
            for elem in result:
                print("Start: " + str(elem["start"]) + " End: " + str(elem["end"]) + " Text: " + str(elem["text"]))
            return result
        
        result = transcribe_audio_whisper(filepath)
        output_file_path = os.path.join(directory_save_file, (filename + "_asr.csv"))
        create_output_file(result, output_file_path)

        return filepath, output_file_path



if __name__ == "__main__":
    cfg = {
            "type": "whisper",
            "model": "small.en",
            "sr": 16000,
            "filepath": "/home/comp/Рабочий стол/AutoDub/output/vad/1_mono_speech_resampled/1_mono_speech_resampled.wav",
            "boundaries": "/home/comp/Рабочий стол/AutoDub/output/vad/1_mono_speech_resampled/1_mono_speech_resampled_boundaries.txt",
            "output_dir": "/home/comp/Рабочий стол/AutoDub/output/asr2",
            "checkpoint_path": "/home/comp/Рабочий стол/AutoDub/checkpoints/asr/whisper"
            }
    inference_asr_wtime(cfg)