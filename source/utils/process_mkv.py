import torch
import torchaudio as ta
import torch.nn.functional as F
import os
from typing import Tuple
import librosa
import pandas as pd
import numpy as np
from moviepy.editor import VideoFileClip, AudioFileClip
from pydub import AudioSegment
from tqdm import tqdm
from speechbrain.inference.enhancement import SpectralMaskEnhancement, WaveformEnhancement


def process_mkv_file(filepath, output_dir):
    return 0


def process_mkv_dir(filepath, output_dir):
    return 0


if __name__ == "__main__":
    cfg = {
        "filepath" : "",
        "output_dir" : ""
    }

    process_mkv_file(**cfg)