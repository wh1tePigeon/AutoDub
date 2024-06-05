import torch
import torchaudio as ta
import torch.nn.functional as F
import os
from typing import Tuple
import librosa
import pandas as pd
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
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from hydra.utils import instantiate
from scipy.spatial.distance import cdist
import numpy as np
from source.utils.util import prepare_device
from source.utils.fader import OverlapAddFader
from omegaconf import OmegaConf


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
                            clean_csv_subs_path = remove_dialogues_n_small_segments(csv_subs_path)
                            _, new_csv_subs_path = cut_n_save(savepath_speech, savepath, clean_csv_subs_path)
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
                        try:
                            segment, sr = ta.load(segment_path)
                            embedding = classifier.encode_batch(segment).squeeze().cpu().numpy()

                            filename = segment_path.split(".")[0].split("/")[-1]
                            savepath = os.path.join(output_dir, (filename + "_embd.npy"))
                            np.save(savepath, embedding, allow_pickle=False)

                            df.at[i, "embd_path"] = savepath

                        except:
                            pass

                df.to_csv(csv_segments_filepath, sep=';', index=False, encoding='utf-8')
              
    return metafilepath


def compute_average_embd(dirpath, output_dir=None):
    assert os.path.exists(dirpath)

    if output_dir is None:
        output_dir = dirpath

    os.makedirs(output_dir, exist_ok=True)

    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
    embds = []

    for filename in tqdm(os.listdir(dirpath), total=len(os.listdir(dirpath))):
        if filename.endswith(".wav"):
            filepath = os.path.join(dirpath, filename)
            try:
                audio, sr = ta.load(filepath)
                embedding = classifier.encode_batch(audio).squeeze().cpu()
                embedding = embedding / embedding.norm()
                embds.append(embedding)
            except:
                pass
    
    average = torch.stack(embds).mean(dim=0).numpy()
    filename = dirpath.split(".")[0].split("/")[-1]
    savepath = os.path.join(output_dir, (filename + "_average_embd.npy"))
    np.save(savepath, average, allow_pickle=False)

    return savepath


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
                        embd = embd / embd.norm()
                        embds.append(embd)

                av = torch.from_numpy(np.load("/home/comp/Рабочий стол/bk/bk_average_embd.npy"))
                av = av / av.norm()
                sims = []
                similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
                for i in range(0, len(embds)):
                    s = similarity(embds[i], av).item()
                    if s >= 0.82:
                        print(i)
                        sims.append(i)

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
#[0.3693041503429413, 0.43667829036712646, 0.593411922454834, 0.6331155300140381, 0.4492175579071045, 0.31460416316986084, 0.7370620965957642, 0.6188938021659851, 0.5243847966194153, 0.6632072329521179, 0.8274864554405212, 0.7624768018722534, 0.7797215580940247, 0.5962139368057251, 0.6379070281982422, 0.7393325567245483, 0.7512869834899902, 0.7231794595718384, 0.5538026690483093, 0.725588858127594, 0.5609574317932129, 0.4644603431224823, 0.5893489122390747, 0.7249554395675659, 0.6515954732894897, 0.607807457447052, 0.7581999897956848, 0.6690129637718201, 0.6436741352081299, 0.7322248220443726, 0.6074910163879395, 0.6675061583518982, 0.715245246887207, 0.5577623248100281, 0.5378882884979248, 0.5732212066650391, 0.6923876404762268, 0.6241010427474976, 0.5194295048713684, 0.42733556032180786, 0.5472899675369263, 0.5297255516052246, 0.4830099046230316, 0.5568930506706238, 0.6803110241889954, 0.6411834955215454, 0.7649610638618469, 0.490192174911499, 0.35297560691833496, 0.6389533281326294, 0.656097412109375, 0.5047973990440369, 0.36591821908950806, 0.3938295245170593, 0.5434467792510986, 0.4685512185096741, 0.6699821949005127, 0.8642969131469727, 0.6297295689582825, ...]
#[0.25111374258995056, 0.6367850303649902, 0.4793241024017334, 0.45974671840667725, 0.2733364701271057, 0.4527074694633484, 0.7471623420715332, 0.49548104405403137, 0.558784544467926, 0.3660196363925934, 0.80808424949646, 0.6894175410270691, 0.5065523386001587, 0.5338807106018066, 0.6331297755241394, 0.6915916204452515, 0.5722323656082153, 0.5078168511390686, 0.7348804473876953]
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

    new_lines = []
    tmp = {
        "start" : None,
        "end" : None,
        "text" : None
    }

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Merging segments to sentences"):
        text = row["text"]
        if tmp["text"] is None:
            tmp["text"] = text
            tmp["start"] = row["start"]
            tmp["end"] = row["end"]

        else: 
            if not tmp["text"].endswith((".", "!", "?")) and text[0].islower():
                tmp["text"] = tmp["text"] + " " + text
                tmp["end"] = row["end"]
            else:
                new_lines.append(tmp)
                tmp = {
                    "start" : None,
                    "end" : None,
                    "text" : None
                }

    new_df = pd.DataFrame(new_lines)

    for i, row in tqdm(new_df.iterrows(), total=len(new_df.index), desc="Removing dialogues"):
        text = row["text"]
        if text[0] == "-":
            new_df = new_df.drop(i)

                
    for i, row in tqdm(new_df.iterrows(), total=len(new_df.index), desc="Deleting short segments"):
        start = row["start"]
        end = row["end"]
        duration = end - start
        if duration < 0.5:
            new_df = new_df.drop(i)
        
    
    csvname = csv_filepath.split(".")[0].split("/")[-1]
    savepath = os.path.join(output_dir, (csvname + "_clean.csv"))
    new_df.to_csv(savepath, sep=';', index=False, encoding='utf-8')

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


def denoise_w_bsrnn(dirpath, cfg, output_dir=None):
    assert os.path.exists(dirpath)

    if output_dir is None:
        output_dir = dirpath

    os.makedirs(output_dir, exist_ok=True)

    #device, device_ids = prepare_device(cfg["n_gpu"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    arch = OmegaConf.load(cfg["model"])
    model = instantiate(arch)
    model = model.to(device)
    # if len(device_ids) > 1:
    #     model = torch.nn.DataParallel(model, device_ids=device_ids)

    checkpoint = torch.load(cfg["checkpoint_path"], map_location=device)
    state_dict = checkpoint["state_dict"]
    model.load_state_dict(state_dict)
    model.eval()
    sr = cfg["sr"]

    for filename in tqdm(os.listdir(dirpath), total=len(os.listdir(dirpath))):
        if filename.endswith(".wav"):
            filepath = os.path.join(dirpath, filename)
            try:
                audio, fs = ta.load(filepath)
                if fs != sr:
                    audio = ta.functional.resample(audio, fs, sr)

                if audio.shape[0] > 1:
                    audio = torch.mean(audio, dim=0, keepdim=True)
        
                # move audio to gpu
                audio = audio.to(device)

                with torch.inference_mode():
                    
                    def forward(audio):
                        _, output = model({"audio": {"mixture": audio},})
                        return output["audio"]
                    
                    if audio.shape[-1] / sr > 5:
                        speech_segments = []
                        segment_len = sr * 5
                        amount_of_segments = audio.shape[-1] // segment_len
                        for i in tqdm(range(amount_of_segments), total=amount_of_segments):
                            start = i * segment_len
                            segment = audio[..., start : start + segment_len]
                            segmet = segment.unsqueeze(0)
                            output = forward(segmet)
                            speech = output["speech"]
                            speech = speech.reshape(1, -1)
                            speech = speech.to("cpu")
                            speech_segments.append(speech)
                        
                        final_segment_l = audio.shape[-1] - amount_of_segments * segment_len
                        if final_segment_l > 0:
                            segment = audio[..., -final_segment_l:]
                            segmet = segment.unsqueeze(0)
                            output = forward(segmet)
                            speech = output["speech"]
                            speech = speech.reshape(1, -1)
                            speech = speech.to("cpu")
                            speech_segments.append(speech)
                        
                        speech = torch.cat(speech_segments, dim=-1)
                        # fader = OverlapAddFader(window_type=cfg["window_type"],
                        #                         chunk_size_second=cfg["chunk_size_second"],
                        #                         hop_size_second=cfg["hop_size_second"],
                        #                         fs=sr,
                        #                         batch_size=cfg["batch_size"])
                        
                        # output = fader(audio, lambda a: forward(a))
                        
                    else:
                        audio = audio.unsqueeze(0)
                        output = forward(audio)
                    
                        speech = output["speech"]
                        speech = speech.reshape(1, -1)
                        speech = speech.to("cpu")

                    savepath = os.path.join(output_dir, filename)
                    ta.save(savepath, speech, sr)
            except:
                pass

    return output_dir





if __name__ == "__main__":
    cfg = {
        "metafilepath" : "/home/comp/Рабочий стол/AutoDub/output/ffmpeg/data.json",
        "output_dir" : "/home/comp/Рабочий стол/AutoDub/output/dataset"
    }
    meta_path = "/home/comp/Рабочий стол/AutoDub/output/dataset/data.json"
    csv_path = "/home/comp/Рабочий стол/AutoDub/output/dataset/sherlock-1/eng/sherlock-1_audio_2_eng_speech/test.csv"
    #merge_sim_segments(meta_path)
    #remove_dialogues_n_small_segments(csv_path)
    #compute_embeddings(meta_path)
    label_embds(meta_path)
    #group_by_label(meta_path)
    #process_data_dir(**cfg)

    config_dict = {
        "n_gpu": 1,
        "model": "/home/comp/Рабочий стол/AutoDub/source/configs/bsrnn/arch/model_conf.yaml",
        "sr": 44100,
        "checkpoint_path": "/home/comp/Рабочий стол/AutoDub/checkpoints/bsrnn/main.pth",
        "window_type": "hann",
        "chunk_size_second": 6.0,
        "hop_size_second": 0.5,
        "batch_size": 4
    }

    dirpath = "/home/comp/Рабочий стол/AutoDub/output/dataset/sherlock-1/eng/sherlock-1_audio_2_eng_speech/segments"
    dirpath2 = "/home/comp/Рабочий стол/AutoDub/output/dataset/sherlock-1/eng/sherlock-1_audio_2_eng_speech/test"
    #output_dir = "/home/comp/Рабочий стол/AutoDub/output/dataset/sherlock-1/eng/test2"
    #denoise_w_bsrnn(dirpath2, config_dict)#, output_dir)
    #compute_average_embd("/home/comp/Рабочий стол/bk")
    