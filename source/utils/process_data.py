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
from source.utils.process_audio import cut_n_save

from speechbrain.inference.speaker import EncoderClassifier
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler


# def extract_speech_track(metafilepath, output_dir):
#     assert os.path.exists(metafilepath)

#     os.makedirs(output_dir, exist_ok=True)

#     with open(metafilepath, 'r', encoding='utf-8') as file:
#         metadata = json.load(file)

#     for i, file in enumerate(metadata["files"]):
#         for language in file["languages"]:
#             tmp = file["languages"][language]
#             if len(tmp["audio_paths"]) >= 1 and len(tmp["subs_paths"]) >= 1:
#                 audio_path = tmp["audio_paths"][0]
                
#                 if os.path.exists(audio_path):
#                     audio, sr = ta.load(audio_path)

#                     if audio.shape[0] == 6:
#                         # in wav files third channel is central sound
#                         speech = audio[2].unsqueeze(0)

#                         filename = audio_path.split(".")[0].split("/")[-1] + "_speech.wav"
#                         savepath = os.path.join(output_dir, filename)
#                         print("Saving " + filename)
#                         ta.save(savepath, speech, sample_rate=sr)
#                         metadata["files"][i]["languages"][language]["speech_path"] = savepath

#     meta_savepath = os.path.join(output_dir, "data.json")
#     with open(meta_savepath, 'w', encoding='utf-8') as f:
#         json.dump(metadata, f, ensure_ascii=False, indent=4)

#     return meta_savepath


# def extract_speech_segments(metafilepath, output_dir):
#     assert os.path.exists(metafilepath)

#     os.makedirs(output_dir, exist_ok=True)

#     with open(metafilepath, 'r', encoding='utf-8') as file:
#         metadata = json.load(file)

#     for i, file in enumerate(metadata["files"]):
#         for language in file["languages"]:
#             tmp = file["languages"][language]
#             if len(tmp["audio_paths"]) >= 1 and len(tmp["subs_paths"]) >= 1:
#                 return 0





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
                            csv_subs_path = srt_to_txt(subs_path, savepath)
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

                for i, row in tqdm(df.iterrows(), total=len(df.index)):
                    segment_path = row["path"]
                    if os.path.exists(segment_path):
                        segment, sr = ta.load(segment_path)
                        embedding = classifier.encode_batch(segment).squeeze().cpu().numpy()

                        filename = segment_path.split(".")[0].split("/")[-1]
                        savepath = os.path.join(output_dir, (filename + "_embd"))
                        np.save(savepath, embedding, allow_pickle=False)

                        df.at[i, "embd_path"] = savepath

                df.to_csv(csv_segments_filepath, sep=';', index=False, encoding='utf-8')
              
    return metafilepath


if __name__ == "__main__":
    cfg = {
        "metafilepath" : "/home/comp/Рабочий стол/AutoDub/output/ffmpeg/data.json",
        "output_dir" : "/home/comp/Рабочий стол/AutoDub/output/dataset"
    }

    compute_embeddings("/home/comp/Рабочий стол/AutoDub/output/dataset/data.json")

    #process_data_dir(**cfg)