import torch
import torchaudio as ta
import torch.nn.functional as F
import os
from typing import Tuple
import librosa
import pandas as pd
import numpy as np
from tqdm import tqdm
import ffmpeg
import json
import pysrt
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from source.utils.process_mkv import process_mkv_dir


#def extract_speech_track(filepath, )


def process_data_dir(dirpath, output_dir):
    assert os.path.exists(dirpath)

    mkv_path = os.path.join(output_dir, "processed_mkv")
    os.makedirs(mkv_path, exist_ok=True)

    #extracting different languages
    meta_savepath, meta = process_mkv_dir(dirpath, mkv_path)

    #extracting speech tracks
    for file in meta.files:
        for language in file.languages:


