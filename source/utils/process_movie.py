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
    #process_dir_dubbed(**cfg)
    
    #path = "/home/comp/Загрузки/Harry Potter and the Chamber of Secrets [Extended]/Audio/ORIGINAL.dts"
    # path1 = "/home/comp/Рабочий стол/RaM/output_1st_track.wav"
    # path2 = "/home/comp/Рабочий стол/RaM/rm1.mkv.part"
    # audio, sr = ta.load(path1)
    # segment = audio[...,:sr * 30]
    # save_path= "/home/comp/Рабочий стол/RaM/segment.wav"
    # ta.save(save_path, segment, sample_rate=sr)

    path = "/home/comp/Рабочий стол/RaM/segment.wav"
    dirpath = "/home/comp/Рабочий стол/RaM"
    segment, sr = ta.load(path)

    for i in range(segment.shape[0]):
        mono = segment[i]
        mono = mono.reshape(1, -1)
        mono_save_path = os.path.join(dirpath, ("segment_" + str(i) + ".wav"))
        ta.save(mono_save_path, mono, sample_rate=sr)

    #print(audio.shape)
    print(sr)

   # ffmpeg -i "/home/comp/Рабочий стол/ldr/ldr.mkv" -map 0:a:1 "/home/comp/Рабочий стол/RaM/output_1st_track.wav"
