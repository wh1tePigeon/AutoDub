import torch
import torchaudio as ta
import torch.nn.functional as F
import os
from typing import Tuple
import pandas as pd
from moviepy.editor import VideoFileClip, AudioFileClip
from pydub import AudioSegment
from tqdm import tqdm


def process_dir_dubbed(dirpath, output_dir, lang_src="eng", lang_trgt="ru", save_separated_video=False):
    assert os.path.exists(dirpath)

    dirpath_src = os.path.join(dirpath, lang_src)
    dirpath_lang_trgt = os.path.join(dirpath, lang_trgt)

    assert os.path.exists(dirpath_src)
    assert os.path.exists(dirpath_lang_trgt)

    src_audio = ""
    src_subs = ""
    trgt_audio = ""
    trgt_subs = ""

    for filename in os.listdir(dirpath_src):
        print("test")




if __name__ == "__main__":
    cfg = {
        "dirpath": "",
        "output_dir": ""
    }
    process_dir_dubbed(**cfg)