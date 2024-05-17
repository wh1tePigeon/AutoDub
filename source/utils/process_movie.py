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



def get_speech(filepath, output_dir, separator_type, separator_cfg):
    assert os.path.exists(filepath)

    separator_types = ["bsrnn_speech", "sb_spectral", "sb_wave"]
    if separator_type not in separator_types:
        raise KeyError

    filename = filepath.split(".")[0].split("/")[-1]
    directory_save_file = os.path.join(output_dir, filename)
    os.makedirs(directory_save_file, exist_ok=True)

    audio, sr = ta.load(filepath)
    if sr != separator_cfg["sr"]:
        audio = ta.functional.resample(audio, sr, separator_cfg["sr"])
        sr = separator_cfg["sr"]

    if separator_type == "bsrnn_speech":
        raise NotImplementedError
    

    elif separator_type == "sb_spectral":
        enhancer = SpectralMaskEnhancement.from_hparams(source="speechbrain/metricgan-plus-voicebank",
            savedir=separator_cfg["checkpoint_path"])
        #lengths = torch.tensor(audio.shape[-1])
        #audio = audio.unsqueeze(0)
        #lengths = lengths.unsqueeze(0).unsqueeze(0)
        audio_clean = enhancer.enhance_file(filepath).unsqueeze(0)
        #audio_clean = enhancer.enhance_file(filepath)

    #loudness_original = librosa.effects.loudness(audio)[0]
    audio = audio.numpy()
    loudness_original = librosa.amplitude_to_db(np.abs(librosa.stft(audio)))
    #loudness_clean = librosa.feature.loudness(audio_clean)[0]

    audio_clean = librosa.util.normalize(audio_clean, target_dB=loudness_original)

    audio = torch.from_numpy(audio)
    background = audio - audio_clean

    speech_save_path = os.path.join(directory_save_file, (filename + "_speech.wav"))
    background_save_path = os.path.join(directory_save_file, (filename + "_background.wav"))
    
    speech = audio_clean.to("cpu")
    background = background.to("cpu")

    ta.save(speech_save_path, speech, sample_rate=sr)
    ta.save(background_save_path, background, sample_rate=sr)

    return [speech_save_path, background_save_path]



if __name__ == "__main__":
    cfg = {
        "filepath": "/home/comp/Рабочий стол/RaM/segment_2_cutted.wav",
        "output_dir": "/home/comp/Рабочий стол/AutoDub/output/enhanced",
        "separator_type" : "sb_spectral",
        "separator_cfg": {
            "sr" : 16000,
            "checkpoint_path" : "/home/comp/Рабочий стол/AutoDub/checkpoints/enhancers/sb_spectral"
        }
    }

    #get_speech(**cfg)

    path_orig = "/home/comp/Рабочий стол/AutoDub/output/enhanced/segment_2_cutted/segment_2_cutted_orig.wav"
    path2_speech = "/home/comp/Рабочий стол/AutoDub/output/enhanced/speech_normalized.wav"

    # from pydub import AudioSegment
    # import numpy as np

    # def calculate_average_volume(segment):
    #     return segment.dBFS

    # # Загрузка аудиофайлов
    # audio1 = AudioSegment.from_file(path_orig)
    # audio2 = AudioSegment.from_file(path2_speech)

    # # Вычисление средней громкости
    # average_volume1 = calculate_average_volume(audio1)
    # average_volume2 = calculate_average_volume(audio2)

    # # Выравнивание громкости
    # if average_volume1 > average_volume2:
    #     audio2 = audio2 + (average_volume1 - average_volume2)
    # elif average_volume1 < average_volume2:
    #     audio1 = audio1 + (average_volume2 - average_volume1)

    # #back = audio1 - audio2

    # # Сохранение результатов
    # #back.export("/home/comp/Рабочий стол/AutoDub/output/enhanced/back_normalized.wav", format="wav")
    # audio2.export("/home/comp/Рабочий стол/AutoDub/output/enhanced/speech_normalized.wav", format="wav")

    orig, sr = ta.load(path_orig)
    speech, sr = ta.load(path2_speech)

    back = orig - speech
    back_path = "/home/comp/Рабочий стол/AutoDub/output/enhanced/back.wav"
    ta.save(back_path, back, sr)


    #process_dir_dubbed(**cfg)
    
    #path = "/home/comp/Загрузки/Harry Potter and the Chamber of Secrets [Extended]/Audio/ORIGINAL.dts"
    # path1 = "/home/comp/Рабочий стол/RaM/output_1st_track.wav"
    # path2 = "/home/comp/Рабочий стол/RaM/rm1.mkv.part"
    # audio, sr = ta.load(path1)
    # segment = audio[...,:sr * 30]
    # save_path= "/home/comp/Рабочий стол/RaM/segment.wav"
    # ta.save(save_path, segment, sample_rate=sr)

    # path = "/home/comp/Рабочий стол/RaM/segment.wav"
    # dirpath = "/home/comp/Рабочий стол/RaM"
    # segment, sr = ta.load(path)

    # for i in range(segment.shape[0]):
    #     mono = segment[i]
    #     mono = mono.reshape(1, -1)
    #     mono_save_path = os.path.join(dirpath, ("segment_" + str(i) + ".wav"))
    #     ta.save(mono_save_path, mono, sample_rate=sr)

    # #print(audio.shape)
    # print(sr)

   # ffmpeg -i "/home/comp/Рабочий стол/ldr/ldr.mkv" -map 0:a:1 "/home/comp/Рабочий стол/RaM/output_1st_track.wav"
