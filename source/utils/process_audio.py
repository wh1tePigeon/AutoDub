import torch
import torchaudio as ta
import os
from typing import Tuple
import pandas as pd
from moviepy.editor import VideoFileClip
from pydub import AudioSegment


def load_n_process_audio(filepath, output_dir, sr) -> Tuple[torch.Tensor, str]:
    assert os.path.exists(filepath)

    audio, fs = ta.load(filepath)
    filename = filepath.split(".")[0].split("/")[-1]
    changed = False

    # resample
    if fs != sr:
        print("Resampling")
        audio = ta.functional.resample(audio, fs, sr)
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
        os.makedirs(directory_save_file, exist_ok=True)

        filename = filename + ".wav"
        filepath = os.path.join(directory_save_file, filename)
        ta.save(filepath, audio, sr)
    
    return [audio, filepath]


def cut_n_save(filepath, output_dir, csv_filepath):
    assert os.path.exists(filepath)
    assert os.path.exists(csv_filepath)

    audio, sr = ta.load(filepath)

    filename = filepath.split(".")[0].split("/")[-1]
    csv_filename = csv_filepath.split(".")[0].split("/")[-1]

    df = pd.read_csv(csv_filepath, delimiter=';', encoding='utf-8')
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
    return filepath, new_csv_path


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


def align_audio_length(csv_filepath, output_dir, filename):
    assert os.path.exists(csv_filepath)

    csv_filename = csv_filepath.split(".")[0].split("/")[-1]
    directory_save_file = os.path.join(output_dir, filename)
    directory_save_file_segments = os.path.join(directory_save_file, "segments")
    os.makedirs(directory_save_file_segments, exist_ok=True)

    df = pd.read_csv(csv_filepath, delimiter=';', encoding='utf-8')

    for i, row in df.iterrows():
        start = row["start"]
        end = row["end"]
        audio_path = row["tts_path"]
        target_len = end - start

        audio = AudioSegment.from_wav(audio_path)

        len = audio.duration_seconds
        speedup = len / target_len
        audio = audio.speedup(playback_speed=speedup)

        segment_name = audio_path.split(".")[0].split("/")[-1]
        save_segment_name = segment_name + "_aligned.wav"
        save_segment_path = os.path.join(directory_save_file_segments, save_segment_name)

        audio.export(save_segment_path, format="wav")
        df.at[i, "aligned_tts"] = save_segment_path

    new_csv_path = os.path.join(directory_save_file, (csv_filename + "_wpaths.csv"))
    df.to_csv(new_csv_path, sep=';', index=False, encoding='utf-8')
    return new_csv_path


if __name__ == "__main__":
    csv_filepath = "/home/comp/Рабочий стол/AutoDub/output/tts/lazy/1_mono_speech_resampled/1_mono_speech_resampled_asr_g_tr_wpaths_tts.csv"
    output_dir = "/home/comp/Рабочий стол/AutoDub/output/aligned_audio"
    filename = "1_mono_speech_resampled"

    align_audio_length(csv_filepath, output_dir, filename)