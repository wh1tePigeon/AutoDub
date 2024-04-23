from omegaconf import DictConfig
import pandas as pd
import os
from googletrans import Translator

def translate(line: str):
    translator = Translator()
    translated_line = translator.translate(line, dest='ru').text
    return translated_line


def process_file(cfg: DictConfig):
    filepath = cfg["input_path"]
    directory_save = cfg["output_dir"]

    assert os.path.exists(filepath)

    filename = filepath.split(".")[0].split("/")[-1]
    
    directory_save_file = os.path.join(directory_save, filename)
    if not os.path.exists(directory_save_file):
            os.mkdir(directory_save_file)

    output = os.path.join(directory_save_file, (filename + "_g_tr.csv"))

    translator = Translator()
    df = pd.read_csv(filepath, delimiter=';', encoding='utf-8')
    df['translation'] = df['transcription'].apply(lambda x: translator.translate(x, dest='ru').text)
    df.to_csv(output, sep=';', index=False, encoding='utf-8')

    return output

if __name__ == "__main__":
    cfg = {
        "input_path" : "/home/comp/Рабочий стол/AutoDub/output/asr/speech_resampled_mono/speech_resampled_mono_asr.csv",
        "output_dir" : "/home/comp/Рабочий стол/AutoDub/output/translated/"
    }
    process_file(cfg)