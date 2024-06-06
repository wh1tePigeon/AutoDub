import os
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
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def get_embeddings(audio_filepath, csv_filepath):
    assert os.path.exists(audio_filepath)
    assert os.path.exists(csv_filepath)

    audio, sr = ta.load(audio_filepath)
    df = pd.read_csv(csv_filepath, delimiter=';', encoding='utf-8')
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
    embeddings = []

    for _, row in tqdm(df.iterrows(), total=len(df.index)):
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


def cluster_with_kmeans(embeddings, n_clusters=None, random_state=0):
    assert n_clusters is not None

    scaler = StandardScaler()
    scaled_embeddings = scaler.fit_transform(embeddings)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state).fit(scaled_embeddings)
    labels = kmeans.labels_
    return labels


def label_speakers(audio_filepath, csv_filepath, output_dir, cluster_type, cluster_cfg: dict):
    _, _, embeddings = get_embeddings(audio_filepath, csv_filepath)
    if cluster_type == "dbscan":
        labels = cluster_with_dbscan(embeddings, **cluster_cfg)

    elif cluster_type == "kmeans":
        labels = cluster_with_kmeans(embeddings, **cluster_cfg)

    df = pd.read_csv(csv_filepath, delimiter=';', encoding='utf-8')
    df["label"] = labels

    csv_filename = csv_filepath.split(".")[0].split("/")[-1]
    directory_save_file = os.path.join(output_dir, csv_filename)
    os.makedirs(directory_save_file, exist_ok=True)

    new_csv_path = os.path.join(directory_save_file, (csv_filename + "_labeled.csv"))
    df.to_csv(new_csv_path, sep=';', index=False, encoding='utf-8')
    return new_csv_path


if __name__ == "__main__":
    cfg = {
        "audio_filepath": "/home/comp/Рабочий стол/ffmpeg/rm2/rm2_audio_1_rus.wav",
        "csv_filepath":  "/home/comp/Рабочий стол/test_out/rm2_subs_3_rus_csv.csv",
        "output_dir": "/home/comp/Рабочий стол/test_out/",
        "cluster_type": "dbscan",
        "cluster_cfg": {
            "metric": "cosine",
            "eps": 0.8,
            "min_samples": 1
        }
        # "cluster_type": "kmeans",
        # "cluster_cfg": {
        #     "n_clusters": 2    
        # }
    }


    #print(cluster_with_dbscan_def(embs))
    #print(cluster_with_dbscan_cosine(embs))
    #print(cluster_with_kmeans(embs, 4))
    label_speakers(**cfg)
    # t2 = "/home/comp/Рабочий стол/AutoDub/output/asr/1_mono_speech_resampled/1_mono_speech_resampled_asr.csv"
    # t1 = "/home/comp/Рабочий стол/AutoDub/output/vad/1_mono_speech_resampled/1_mono_speech_resampled.wav"
    #_, _ , embs = get_embeddings(t1, t2)
