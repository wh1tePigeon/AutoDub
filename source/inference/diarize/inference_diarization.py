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
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np




def get_embeddings(audio_filepath, csv_filepath):
    assert os.path.exists(audio_filepath)
    assert os.path.exists(csv_filepath)

    audio, sr = ta.load(audio_filepath)
    df = pd.read_csv(csv_filepath, delimiter=';', encoding='utf-8')
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
    embeddings = []

    for _, row in df.iterrows():
        start_time = row["start"]
        end_time = row["end"]

        start = int(start_time * sr)
        end = int(end_time * sr)

        segment = audio[..., start:end]
        embedding = classifier.encode_batch(segment).squeeze()
        embeddings.append(embedding)
    
    embeddings = torch.stack(embeddings)
    return [audio_filepath, csv_filepath, embeddings]


def cluster_with_dbscan(embeddings, metric, eps, min_samples):
    clustering = DBSCAN(metric=metric, eps=eps, min_samples=min_samples).fit(embeddings)
    labels = clustering.labels_
    return labels


# def cluster_with_dbscan_def(embeddings, eps=310, min_samples=1):
#     clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(embeddings)
#     labels = clustering.labels_
#     return labels


# def cluster_with_dbscan_cosine(embeddings, eps=0.5, min_samples=1):
#     clustering = DBSCAN(metric='cosine', eps=eps, min_samples=min_samples).fit(embeddings)
#     labels = clustering.labels_
#     return labels


def cluster_with_kmeans(embeddings, n_clusters=None, random_state=0):
    assert n_clusters is not None

    scaler = StandardScaler()
    scaled_embeddings = scaler.fit_transform(embeddings)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state).fit(scaled_embeddings)
    labels = kmeans.labels_
    return labels


def label_speakers(audio_filepath, csv_filepath, cluster_type, cluster_cfg: dict):
    _, _, embeddings = get_embeddings(audio_filepath, csv_filepath)
    if cluster_type == "dbscan":
        labels = cluster_with_dbscan(embeddings, **cluster_cfg)

    elif cluster_type == "kmeans":
        labels = cluster_with_kmeans(embeddings, **cluster_cfg)

        



if __name__ == "__main__":
    cfg = {
        "audio_filepath": "/home/comp/Рабочий стол/AutoDub/output/vad/1_mono_speech_resampled/1_mono_speech_resampled.wav",
        "csv_filepath":  "/home/comp/Рабочий стол/AutoDub/output/asr2/1_mono_speech_resampled/1_mono_speech_resampled_asr.csv"
    }

    _, _, embs = get_embeddings(**cfg)


    print(cluster_with_dbscan_def(embs))
    print(cluster_with_dbscan_cosine(embs))
    print(cluster_with_kmeans(embs, 4))





# from pyannote.audio import Pipeline
# pipeline = Pipeline.from_pretrained(
#     "pyannote/speaker-diarization-3.1",
#     use_auth_token="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")


# # apply pretrained pipeline
# diarization = pipeline("/home/comp/Рабочий стол/AutoDub/output/vad/1_mono_speech_resampled/1_mono_speech_resampled.wav")

# # print the result
# for turn, _, speaker in diarization.itertracks(yield_label=True):
#     print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")