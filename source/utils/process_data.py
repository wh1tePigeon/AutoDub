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
import shutil
import pysrt
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from source.utils.process_mkv import process_mkv_dir, srt_to_csv
from source.utils.process_audio import cut_n_save

from speechbrain.inference.speaker import EncoderClassifier
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler

from scipy.spatial.distance import cdist
import numpy as np


def process_data_dir(metafilepath, output_dir):
    assert os.path.exists(metafilepath)

    with open(metafilepath, 'r', encoding='utf-8') as file:
        metadata = json.load(file)

    for i, file in enumerate(metadata["files"]):
        mvkfilename = file["mkvfilepath"].split(".")[0].split("/")[-1]
        filedir = os.path.join(output_dir, mvkfilename)

        for language in file["languages"]:
            savepath = os.path.join(filedir, language)
            os.makedirs(savepath, exist_ok=True)
            tmp = file["languages"][language]

            if len(tmp["audio_paths"]) >= 1 and len(tmp["subs_paths"]) >= 1:
                audio_path = tmp["audio_paths"][0]

                if os.path.exists(audio_path):
                    audio, sr = ta.load(audio_path)

                    if audio.shape[0] == 6:
                        # in wav files third channel is central sound
                        speech = audio[2].unsqueeze(0)

                        speech_filename = audio_path.split(".")[0].split("/")[-1] + "_speech.wav"
                        savepath_speech = os.path.join(savepath, speech_filename)
                        print("Saving " + speech_filename)
                        ta.save(savepath_speech, speech, sample_rate=sr)
                        del speech
                        del audio
                        metadata["files"][i]["languages"][language]["speech_path"] = savepath_speech

                        subs_path = tmp["subs_paths"][0]
                        if os.path.exists(subs_path):
                            csv_subs_path = srt_to_csv(subs_path, savepath)
                            _, new_csv_subs_path = cut_n_save(savepath_speech, savepath, csv_subs_path)
                            metadata["files"][i]["languages"][language]["csv_w_segments"] = new_csv_subs_path



    meta_savepath = os.path.join(output_dir, "data.json")
    with open(meta_savepath, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)

    return meta_savepath


def compute_embeddings(metafilepath):
    assert os.path.exists(metafilepath)

    with open(metafilepath, 'r', encoding='utf-8') as file:
        metadata = json.load(file)

    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

    for i, file in enumerate(metadata["files"]):
        for language in file["languages"]:
            tmp = file["languages"][language]
            if "csv_w_segments" in tmp:
                csv_segments_filepath = tmp["csv_w_segments"]

                output_dir = os.path.join(os.path.dirname(csv_segments_filepath), "embds")
                os.makedirs(output_dir, exist_ok=True)

                df = pd.read_csv(csv_segments_filepath, delimiter=';', encoding='utf-8')

                print("Computing embds for " + tmp["speech_path"].split(".")[0].split("/")[-1])
                for i, row in tqdm(df.iterrows(), total=len(df.index)):
                    segment_path = row["path"]
                    if os.path.exists(segment_path):
                        segment, sr = ta.load(segment_path)
                        embedding = classifier.encode_batch(segment).squeeze().cpu().numpy()

                        filename = segment_path.split(".")[0].split("/")[-1]
                        savepath = os.path.join(output_dir, (filename + "_embd.npy"))
                        np.save(savepath, embedding, allow_pickle=False)

                        df.at[i, "embd_path"] = savepath

                df.to_csv(csv_segments_filepath, sep=';', index=False, encoding='utf-8')
              
    return metafilepath


def cosine_distance(X1, X2):
    return cdist(X1, X2, 'cosine')[0][0]


class DBSCANCosine:
    def __init__(self, eps=0.6, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
    
    def fit(self, X):
        n_points = len(X)
        labels = np.zeros(n_points, dtype=np.int32)
        cluster_id = 0
        
        for point_index in range(n_points):
            if labels[point_index]!= 0:
                continue
            
            neighbors = []
            for neighbor_index in range(n_points):
                if point_index == neighbor_index:
                    continue
                
                distance = cosine_distance(X[point_index].reshape(1, -1), X[neighbor_index].reshape(1, -1))
                
                if distance > self.eps:
                    neighbors.append(neighbor_index)
            
            if len(neighbors) >= self.min_samples:
                labels[neighbors] = cluster_id
                cluster_id += 1

                # for neighbor_index in neighbors:
                #     if labels[neighbor_index] == 0:
                #         new_neighbors = [n for n in neighbors if cosine_distance(X[neighbor_index].reshape(1, -1), X[n].reshape(1, -1)) > self.eps]
                #         if len(new_neighbors) >= self.min_samples:
                #             labels[new_neighbors] = cluster_id
                #             cluster_id += 1
        return labels
                            


def label_embds(metafilepath):
    assert os.path.exists(metafilepath)

    with open(metafilepath, 'r', encoding='utf-8') as file:
        metadata = json.load(file)

    for i, file in enumerate(metadata["files"]):
        for language in file["languages"]:
            tmp = file["languages"][language]
            if "csv_w_segments" in tmp:
                csv_segments_filepath = tmp["csv_w_segments"]

                df = pd.read_csv(csv_segments_filepath, delimiter=';', encoding='utf-8')

                print("Labeling embds for " + tmp["speech_path"].split(".")[0].split("/")[-1])
                embds = []
                for i, row in tqdm(df.iterrows(), total=len(df.index)):
                    embd_path = row["embd_path"]
                    if os.path.exists(embd_path):
                        embd = torch.from_numpy(np.load(embd_path))
                        #embd = embd / embd.norm()
                        embds.append(embd)

                embds = torch.stack(embds)
                #clustering = DBSCAN(metric="cosine", eps=0.4, min_samples=10).fit(embds)
                #labels = clustering.labels_
                #labels = DBSCANCosine(eps=0.7, min_samples=15).fit(embds)
                scaler = StandardScaler()
                scaled_embeddings = scaler.fit_transform(embds)

                kmeans = KMeans(n_clusters=16, random_state=0).fit(scaled_embeddings)
                labels = kmeans.labels_

                df = df.assign(label=labels)
                df.to_csv(csv_segments_filepath, sep=';', index=False, encoding='utf-8')
              
    return metafilepath


def merge_sim_segments(metafilepath):
    assert os.path.exists(metafilepath)

    with open(metafilepath, 'r', encoding='utf-8') as file:
        metadata = json.load(file)

    for i, file in enumerate(metadata["files"]):
        for language in file["languages"]:
            tmp = file["languages"][language]
            if "csv_w_segments" in tmp:
                csv_segments_filepath = tmp["csv_w_segments"]

                df = pd.read_csv(csv_segments_filepath, delimiter=';', encoding='utf-8')

                print("Merging similar segments for " + tmp["speech_path"].split(".")[0].split("/")[-1])
                embds = []
                for i, row in tqdm(df.iterrows(), total=len(df.index)):
                    embd_path = row["embd_path"]
                    if os.path.exists(embd_path):
                        embd = torch.from_numpy(np.load(embd_path))
                        embd = embd / embd.norm()
                        embds.append(embd)

                sims = []
                similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
                for i in range(0, len(embds) - 1):
                    s = similarity(embds[i], embds[i + 1]).item()
                    print(s)
                    sims.append(s)
                
                print("1")


                #embds = torch.stack(embds)
                #clustering = DBSCAN(metric="cosine", eps=0.4, min_samples=10).fit(embds)
                #labels = clustering.labels_
                #labels = DBSCANCosine(eps=0.7, min_samples=15).fit(embds)
                # scaler = StandardScaler()
                # scaled_embeddings = scaler.fit_transform(embds)

                # kmeans = KMeans(n_clusters=16, random_state=0).fit(scaled_embeddings)
                # labels = kmeans.labels_

                # df = df.assign(label=labels)
                # df.to_csv(csv_segments_filepath, sep=';', index=False, encoding='utf-8')
              
    return metafilepath


def remove_dialogues_n_small_segments(csv_filepath, output_dir=None):
    assert os.path.exists(csv_filepath)

    if output_dir is None:
        output_dir = os.path.dirname(csv_filepath)

    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_filepath, delimiter=';', encoding='utf-8')

    for i, row in tqdm(df.iterrows(), total=len(df.index)):
        text = row["text"]
        if text[0] == "-":
            df = df.drop(i)
        
        start = row["start"]
        end = row["end"]
        duration = end - start
        if duration < 0.5:
            df = df.drop(i)
    
    csvname = csv_filepath.split(".")[0].split("/")[-1]
    savepath = os.path.join(output_dir, (csvname + "_clean.csv"))
    df.to_csv(savepath, sep=';', index=False, encoding='utf-8')

    return savepath


def group_by_label(metafilepath):
    assert os.path.exists(metafilepath)

    with open(metafilepath, 'r', encoding='utf-8') as file:
        metadata = json.load(file)

    for i, file in enumerate(metadata["files"]):
        for language in file["languages"]:
            tmp = file["languages"][language]
            if "csv_w_segments" in tmp:
                csv_segments_filepath = tmp["csv_w_segments"]

                output_dir = os.path.join(os.path.dirname(csv_segments_filepath), "labels")

                df = pd.read_csv(csv_segments_filepath, delimiter=';', encoding='utf-8')

                print("Grouping segments by label for " + tmp["speech_path"].split(".")[0].split("/")[-1])
                for i, row in tqdm(df.iterrows(), total=len(df.index)):
                    label = row["label"]
                    segment_path = row["path"]
                    label_savedir = os.path.join(output_dir, str(label))
                    os.makedirs(label_savedir, exist_ok=True)
                    file_savepath = os.path.join(label_savedir, segment_path.split("/")[-1])
                    shutil.copy2(segment_path, file_savepath)

              
    return metafilepath



if __name__ == "__main__":
    cfg = {
        "metafilepath" : "/home/comp/Рабочий стол/AutoDub/output/ffmpeg/data.json",
        "output_dir" : "/home/comp/Рабочий стол/AutoDub/output/dataset"
    }
    meta_path = "/home/comp/Рабочий стол/AutoDub/output/dataset/data.json"
    csv_path = "/home/comp/Рабочий стол/AutoDub/output/dataset/sherlock-1/eng/sherlock-1_audio_2_eng_speech/test.csv"
    #merge_sim_segments(meta_path)
    remove_dialogues_n_small_segments(csv_path)
    #compute_embeddings(meta_path)
    #label_embds(meta_path)
    #group_by_label(meta_path)
    #process_data_dir(**cfg)