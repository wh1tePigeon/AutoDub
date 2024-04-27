import os
import torch
import torchaudio as ta
import pandas as pd
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts


def lazy_tts(csv_filepath, output_dir, filename, checkpoint_path):
    assert os.path.exists(csv_filepath)

    config = XttsConfig()
    config_path = os.path.join(checkpoint_path, "config.json")
    config.load_json(config_path)
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=checkpoint_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    directory_save_file = os.path.join(output_dir, filename)
    directory_save_file_segments = os.path.join(directory_save_file, "segments")
    os.makedirs(directory_save_file_segments, exist_ok=True)

    csv_filename = csv_filepath.split(".")[0].split("/")[-1]
    df = pd.read_csv(csv_filepath, delimiter=';', encoding='utf-8')

    for i, row in df.iterrows():
        refer_wav_path = row["path"]
        assert os.path.exists(refer_wav_path)
        text = row["translation"]

        print("Processing segment " + str(i))
        gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=refer_wav_path)

        out = model.inference(
            text,
            "ru",
            gpt_cond_latent,
            speaker_embedding,
            temperature=0.7,
        )
        sr = 24000
        audio = torch.tensor(out["wav"]).unsqueeze(0)

        segment_filename = refer_wav_path.split(".")[0].split("/")[-1]
        segment_filename = segment_filename + "_tts.wav"

        save_file_path = os.path.join(directory_save_file_segments, segment_filename)
        df.at[i, "tts_path"] = save_file_path

        ta.save(save_file_path, audio, sample_rate=sr)

    new_csv_path = os.path.join(directory_save_file, (csv_filename + "_tts.csv"))
    df.to_csv(new_csv_path, sep=';', index=False, encoding='utf-8')
    return new_csv_path



if __name__ == "__main__":
    csv_filepath = "/home/comp/Рабочий стол/AutoDub/output/cutted/1_mono_speech_resampled/1_mono_speech_resampled_asr_g_tr_wpaths.csv"
    output_dir = "/home/comp/Рабочий стол/AutoDub/output/tts/lazy"
    filename = "1_mono_speech_resampled"
    checkpoint_path = "/home/comp/Рабочий стол/AutoDub/checkpoints/tts/lazy"
    lazy_tts(csv_filepath, output_dir, filename, checkpoint_path)