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
from source.utils.process_mkv import process_mkv_dir, srt_to_txt


def extract_speech_track(metafilepath, output_dir):
    assert os.path.exists(metafilepath)

    os.makedirs(output_dir, exist_ok=True)

    with open(metafilepath, 'r', encoding='utf-8') as file:
        metadata = json.load(file)

    for i, file in enumerate(metadata["files"]):
        for language in file["languages"]:
            tmp = file["languages"][language]
            if len(tmp["audio_paths"]) >= 1 and len(tmp["subs_paths"]) >= 1:
                audio_path = tmp["audio_paths"][0]
                
                if os.path.exists(audio_path):
                    audio, sr = ta.load(audio_path)

                    if audio.shape[0] == 6:
                        # in wav files third channel is central sound
                        speech = audio[2].unsqueeze(0)

                        filename = audio_path.split(".")[0].split("/")[-1]
                        savepath = os.path.join(output_dir, (filename + "_speech.wav"))
                        ta.save(savepath, speech, sample_rate=sr)
                        metadata["files"][i]["languages"][language]["speech_path"] = savepath

    meta_savepath = os.path.join(output_dir, "data.json")
    with open(meta_savepath, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)

    return meta_savepath


def process_data_dir(cfg):
    assert os.path.exists(cfg.dirpath)

    mkv_path = os.path.join(cfg.output_dir, "processed_mkv")
    os.makedirs(mkv_path, exist_ok=True)

    #extracting different languages
    meta_savepath, meta = process_mkv_dir(cfg.dirpath, mkv_path, languages=["eng", "rus"])

    files_metadata = []
    #extracting speech tracks
    for file in meta.files:
        meta = {"mkvfilepath" : file.mkvfilepath}
        for language in file.languages:
            if len(language.audio_paths) >= 1 and len(language.subs_paths):
                audio_path = language.audio_paths[0]
                srt_path = language.subs_paths[0]
                output_dir_speech = os.path.join(cfg.output_dir, "speech")
                speech_savepath = extract_speech_track(audio_path, output_dir_speech)
                srt_csv_savepath = srt_to_txt(srt_path, output_dir_speech)
                meta[""]
                #data = 


if __name__ == "__main__":
    cfg = {
        "metafilepath" : "/home/comp/Рабочий стол/ffmpeg/data.json",
        "output_dir" : "/home/comp/Рабочий стол/AutoDub/output/dataset/speech"
    }

    extract_speech_track(**cfg)