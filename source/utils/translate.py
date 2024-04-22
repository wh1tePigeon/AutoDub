from omegaconf import DictConfig
import pandas as pd
import os
from googletrans import Translator

def translate(line: str):
    translator = Translator()
    translated_line = translator.translate(line, dest='ru').text
    return translated_line


def process_file(cfg: DictConfig):
    file_path = cfg["input_file_path"]
    save_dir = cfg["output_dir_path"]

    #assert os.path.exists(file_path)
    
    #if not os.path.exists(save_dir):
    #    os.mkdir(save_dir)




    translator = Translator()
    df = pd.read_csv(file_path, delimiter=';', encoding='utf-8')
    df['translation'] = df['transcription'].apply(lambda x: translator.translate(x, dest='ru').text)
    df.to_csv(save_dir, index=False, encoding='utf-8')

if __name__ == "__main__":
    cfg = {
        "input_file_path" : "/home/comp/Рабочий стол/AutoDub/input/asr_result.csv",
        "output_dir_path" : "/home/comp/Рабочий стол/AutoDub/output/translated/asr_result.csv"
    }
    process_file(cfg)