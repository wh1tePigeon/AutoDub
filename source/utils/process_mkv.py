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
import ffmpeg
from speechbrain.inference.enhancement import SpectralMaskEnhancement, WaveformEnhancement
from moviepy.editor import VideoFileClip


def process_mkv_file(filepath, output_dir, languages, extract_srt=True, extract_video=False):
    assert os.path.exists(filepath)
    
    filename = filepath.split(".")[0].split("/")[-1]
    os.makedirs(output_dir, exist_ok=True)

    stream_info = ffmpeg.probe(filepath)
    tracks_info = stream_info['streams']

    audio_tracks = []

    if extract_srt:
        subs_tracks = []

    if extract_video:
        video_tracks = []
    
    for track in tracks_info:
        if track["codec_type"] == "audio":
            if track["tags"]["language"] in languages:
                audio_tracks.append(track)
        
        if extract_video and track["codec_type"] == "video":
            video_tracks.append(track)

        if extract_srt and track["codec_type"] == "subtitle":
            if track["tags"]["language"] in languages:
                subs_tracks.append(track)

    for track in audio_tracks:
        track_filename = filename + "_" + track["tags"]["language"] + ".wav"
        stream = ffmpeg.input(filepath, ss=None, t=-1, select_streams=f'[stream={track["index"]}]:kind={track["codec_type"]}')
        ffmpeg.output(stream, track_filename).run()


    #return audios
    
    
    #stream1 = ffmpeg.input(filepath)
    return 0


def process_mkv_dir(filepath, output_dir):
    return 0


if __name__ == "__main__":
    cfg = {
        "filepath" : "/home/comp/Рабочий стол/RaM/rm1.mkv",
        "languages": ["eng", "rus"],
        "output_dir" : "/home/comp/Рабочий стол/ffmpeg"
    }

    process_mkv_file(**cfg)