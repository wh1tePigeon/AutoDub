import os
import pandas as pd
import ffmpeg
import json
import pysrt


def process_mkv_file(filepath, output_dir, languages, extract_srt=True, extract_video=False):
    assert os.path.exists(filepath)

    filename = filepath.split(".")[0].split("/")[-1]
    os.makedirs(output_dir, exist_ok=True)

    stream_info = ffmpeg.probe(filepath)
    tracks_info = stream_info['streams']

    audio_tracks = []

    if extract_srt:
        subs_tracks = []

    if extract_video:
        video_tracks = []
        
    for track in tracks_info:
        if track["codec_type"] == "audio":
            if track["tags"]["language"] in languages:
                audio_tracks.append(track)
        
        if extract_video and track["codec_type"] == "video":
            video_tracks.append(track)

        if extract_srt and track["codec_type"] == "subtitle":
            if track["tags"]["language"] in languages:
                subs_tracks.append(track)

    ff = ffmpeg.input(filepath)
    video_save_paths = []
    for_each_lang = {lang: {"audio_paths": [], "subs_paths": []} for lang in languages}

    for track in audio_tracks:
        lang = track["tags"]["language"]
        track_filename = filename + "_audio_" + str(track["index"]) + "_" + lang + ".wav"
        track_filepath = os.path.join(output_dir, track_filename)

        for_each_lang[lang]["audio_paths"].append(track_filepath)
        m = "0:" + str(track["index"])
        ff.output(track_filepath, **{"map": m}).run()
    
    if extract_srt:
        for track in subs_tracks:
            lang = track["tags"]["language"]
            track_filename = filename + "_subs_" + str(track["index"]) + "_" + lang + ".srt"
            track_filepath = os.path.join(output_dir, track_filename)
            
            for_each_lang[lang]["subs_paths"].append(track_filepath)
            m = "0:" + str(track["index"])
            ff.output(track_filepath, **{"map": m}).run()

    if extract_video:
        for track in video_tracks:
            track_filename = filename + "_video_" + str(track["index"]) + ".mp4"
            track_filepath = os.path.join(output_dir, track_filename)
            video_save_paths.append(track_filepath)
            m = "0:" + str(track["index"])
            ff.output(track_filepath, **{"map": m}).run()

    return video_save_paths, for_each_lang



def process_mkv_dir(dirpath, output_dir, languages, extract_srt=True, extract_video=False):
    assert os.path.exists(dirpath)

    dir_meta = {"dirpath": str(dirpath)}
    files_metadata = []
    for filename in os.listdir(dirpath):
        filepath = os.path.join(dirpath, filename)
        if os.path.isfile(filepath):
            ext = filename.split(".")[-1]
            if ext == "mkv" or ext == "m4v":
                print("Processing " + filename)
                out_path = os.path.join(output_dir, filename.split(".")[0])
                video_paths, for_each_lang = process_mkv_file(filepath, out_path,
                                                              languages, extract_srt,
                                                              extract_video)
                meta = {"mkvfilepath": filepath}
                meta["video_paths"] = video_paths
                meta["languages"] = for_each_lang
                files_metadata.append(meta)

    dir_meta["files"] = files_metadata
    os.makedirs(output_dir, exist_ok=True)
    meta_savepath = os.path.join(output_dir, "data.json")

    with open(meta_savepath, 'w', encoding='utf-8') as f:
        json.dump(dir_meta, f, ensure_ascii=False, indent=4)
    
    return meta_savepath, dir_meta


def srt_to_csv(filepath, output_dir):
    assert os.path.exists(filepath)

    filename = filepath.split(".")[0].split("/")[-1]
    os.makedirs(output_dir, exist_ok=True)

    subs = pysrt.open(filepath)
    data = []
    for id, subtitle in enumerate(subs):
        start_time = subtitle.start.hours * 3600 + subtitle.start.minutes * 60 + subtitle.start.seconds + subtitle.start.milliseconds / 1000
        end_time = subtitle.end.hours * 3600 + subtitle.end.minutes * 60 + subtitle.end.seconds + subtitle.end.milliseconds / 1000
        #text = subtitle.text
        text = subtitle.text.replace('\n', ' ')
        data.append({
                'id': id,
                'start': start_time,
                'end': end_time,
                'text': text
            })
    df = pd.DataFrame(data)
    savepath = os.path.join(output_dir, (filename + "_csv.csv"))
    df.to_csv(savepath, index=False, sep=";")

    return savepath


if __name__ == "__main__":
    cfg = {
        "filepath" : "/home/comp/Рабочий стол/RaM/rm1.mkv",
        "languages": ["eng", "rus"],
        "output_dir" : "/home/comp/Рабочий стол/ffmpeg"
    }

    cfg2 = {
        "dirpath" : "/home/comp/Рабочий стол/bbt",
        "languages": ["eng", "rus"],
        "output_dir" : "/home/comp/Рабочий стол/AutoDub/output/bbttest2"
    }

    cfg3 = {
        "filepath" : "/home/comp/Рабочий стол/test_inp/rm2_subs_3_rus.srt",
        "output_dir" : "/home/comp/Рабочий стол/test_out/"
    }

    process_mkv_dir(**cfg2)
    #process_mkv_file(**cfg)

    #srt_to_txt(**cfg3)