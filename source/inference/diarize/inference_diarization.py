import os
import csv
import sys
import torch
import torchaudio as ta
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from source.utils.util import prepare_device
from source.utils.process_audio import load_n_process_audio
from speechbrain.inference.speaker import EncoderClassifier
import pandas as pd


def get_embeddings(audio_filepath, csv_filepath):
    assert os.path.exists(audio_filepath)
    assert os.path.exists(csv_filepath)

    audio, sr = ta.load(audio_filepath)
    df = pd.read_csv(csv_filepath, delimiter=';', encoding='utf-8')
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
    embeddings = []

    for index, row in df.iterrows():
        start_time = row["start"]
        end_time = row["end"]

        start = int(start_time * sr)
        end = int(end_time * sr)

        segment = audio[..., start:end]
        embedding = classifier.encode_batch(segment)
        embeddings.append(embedding)
    
    return [audio_filepath, csv_filepath, embeddings]



if __name__ == "__main__":
    cfg = {
        "audio_filepath": "/home/comp/Рабочий стол/AutoDub/output/vad/1_mono_speech_resampled/1_mono_speech_resampled.wav",
        "csv_filepath":  "/home/comp/Рабочий стол/AutoDub/output/asr2/1_mono_speech_resampled/1_mono_speech_resampled_asr.csv"
    }

    get_embeddings(**cfg)
    print("kurwa")





# from pyannote.audio import Pipeline
# pipeline = Pipeline.from_pretrained(
#     "pyannote/speaker-diarization-3.1",
#     use_auth_token="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")


# # apply pretrained pipeline
# diarization = pipeline("/home/comp/Рабочий стол/AutoDub/output/vad/1_mono_speech_resampled/1_mono_speech_resampled.wav")

# # print the result
# for turn, _, speaker in diarization.itertracks(yield_label=True):
#     print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")