import pandas as pd
import os
from googletrans import Translator
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# def translate(line: str):
#     translator = Translator()
#     translated_line = translator.translate(line, dest='ru').text
#     return translated_line


def translate_file(cfg):
    filepath = cfg["filepath"]
    directory_save = cfg["output_dir"]

    assert os.path.exists(filepath)

    filename = filepath.split(".")[0].split("/")[-1]
    directory_save_file = os.path.join(directory_save, filename)
    os.makedirs(directory_save_file, exist_ok=True)

    if cfg["use"] not in ["google", "opusmt"]:
        raise KeyError
    
    output = os.path.join(directory_save_file, (filename + "_tr.csv"))
    df = pd.read_csv(filepath, delimiter=';', encoding='utf-8')

    if cfg["use"] == "google":
        translator = Translator()
        df['translation'] = df['transcription'].apply(lambda x: translator.translate(x, dest='ru').text)
    
    elif cfg["use"] == "opusmt":
        tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ru", cache_dir=cfg["checkpoint_path"])
        model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-ru", cache_dir=cfg["checkpoint_path"])
    
        def translate_opusmt(text):
            inputs = tokenizer.encode(text, return_tensors="pt")
            outputs = model.generate(inputs, max_length=40, num_beams=4, early_stopping=True)
            translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return translated_text

        df['translation'] = df['text'].apply(lambda x: translate_opusmt(x))

    df.to_csv(output, sep=';', index=False, encoding='utf-8')

    return output


if __name__ == "__main__":
    cfg = {
        #"filepath" : "/home/comp/Рабочий стол/AutoDub/output/asr/speech_resampled_mono/speech_resampled_mono_asr.csv",
        "filepath" : "/home/comp/Рабочий стол/AutoDub/output/label/1_mono_speech_resampled_asr/1_mono_speech_resampled_asr_labeled.csv",
        "output_dir" : "/home/comp/Рабочий стол/AutoDub/output/translated/",
        "use": "opusmt",
        "checkpoint_path" : "/home/comp/Рабочий стол/AutoDub/checkpoints/translate/opusmt"
    }
    translate_file(cfg)