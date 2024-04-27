import torch
import torchaudio as ta
import os
from typing import Tuple
import pandas as pd
from moviepy.editor import VideoFileClip


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


def cut_n_save(audio_path, output_dir, csv_path):
    assert os.path.exists(audio_path)
    assert os.path.exists(csv_path)

    audio, sr = ta.load(audio_path)

    filename = audio_path.split(".")[0].split("/")[-1]
    csv_filename = csv_path.split(".")[0].split("/")[-1]

    df = pd.read_csv(csv_path, delimiter=';', encoding='utf-8')
    directory_save_file = os.path.join(output_dir, filename)
    directory_save_file_segments = os.path.join(directory_save_file, "segments")
    os.makedirs(directory_save_file_segments, exist_ok=True)

    for i, row in df.iterrows():
        start = row["start"]
        end = row["end"]
        id = row["id"]

        start = max(int(start * sr), 0)
        end = min(int(end * sr), audio.shape[-1])
        audio_segment = audio[..., start:end]

        save_segment_name = filename + '_' + str(id) + ".wav"
        save_segment_path = os.path.join(directory_save_file_segments, save_segment_name)

        ta.save(save_segment_path, audio_segment, sample_rate=sr)
        df.at[i, "path"] = save_segment_path

    new_csv_path = os.path.join(directory_save_file, (csv_filename + "_wpaths.csv"))
    df.to_csv(new_csv_path, sep=';', index=False, encoding='utf-8')
    return audio_path, new_csv_path


def separate_audio_n_video(filepath, output_dir):
    assert os.path.exists(filepath)

    video_ext = filepath.split(".")[-1]
    filename = filepath.split(".")[0].split("/")[-1]
    directory_save_file = os.path.join(output_dir, filename)
    os.makedirs(directory_save_file, exist_ok=True)

    video = VideoFileClip(filepath)
    audio = video.audio

    audio_save_path = os.path.join(directory_save_file, (filename + "_audio.wav"))
    video_save_path = os.path.join(directory_save_file, (filename + "_video." + video_ext))

    audio.write_audiofile(audio_save_path)
    video.write_videofile(video_save_path, audio=False)

    return [audio_save_path, video_save_path]



#if __name__ == "__main__":
    #filepath = "/home/comp/Рабочий стол/AutoDub/input/w2.mp4"
    #output_dir = "/home/comp/Рабочий стол/AutoDub/output/video_n_audio_separated"

    #separate_audio_n_video(filepath, output_dir)