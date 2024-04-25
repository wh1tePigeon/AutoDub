import torch
import torchaudio as ta
import os
from typing import Tuple

def load_n_process_audio(input_path, output_dir, sr) -> Tuple[torch.Tensor, str]:
    assert os.path.exists(input_path)

    audio, fs = ta.load(input_path)
    filepath = input_path
    filename = input_path.split(".")[0].split("/")[-1]
    changed = False

    # resample
    if fs != sr:
        print("Resampling")
        audio = ta.functional.resample(
        audio,
        fs,
        sr
    )
        filename += "_resampled"
        changed = True
    
    # make audio single channel
    if audio.shape[0] > 1:
        print("Treat as monochannel")
        audio = torch.mean(audio, dim=0, keepdim=True)
        filename += "_mono"
        changed = True

    # save changed audio
    if changed:
        directory_save_file = os.path.join(output_dir, filename)

        #if not os.path.exists(directory_save_file):
        #    os.mkdir(directory_save_file)
        os.makedirs(directory_save_file, exist_ok=True)

        filename = filename + ".wav"
        filepath = os.path.join(directory_save_file, filename)
        ta.save(filepath, audio, sr)
    
    return [audio, filepath]
