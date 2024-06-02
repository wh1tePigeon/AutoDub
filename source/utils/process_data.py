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


def process_data_dir(dirpath, output_dir):
    assert os.path.exists(dirpath)

    os.makedirs(output_dir, exist_ok=True)
    meta_savepath = process_mkv_dir(dirpath, output_dir)

