import os
import csv
import sys
import torch
import torchaudio as ta
import multiprocessing
from hydra.utils import instantiate
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from source.utils.util import prepare_device
from source.utils.process_audio import load_n_process_audio
from source.text_encoder.ctc_char_text_encoder import CTCCharTextEncoder
from omegaconf import OmegaConf
import whisper


# get boundaries
def read_and_process_file(file_path):
    speech_segments = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if parts[3] == "SPEECH":
                start_time = float(parts[1])
                end_time = float(parts[2])
                speech_segments.append((start_time, end_time))
    return speech_segments


# write output file
def create_output_file(speech_segments, transcriptions, output_file_path):
    data = [{'id' : i, 'start': speech_segment[0], 'end': speech_segment[1], 'transcription': transcription}
        for i, speech_segment, transcription in zip(range(len(speech_segments)), speech_segments, transcriptions)]

    with open(output_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['id', 'start', 'end', 'transcription']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=';')

        writer.writeheader()
        writer.writerows(data)


def inference_asr(cfg):
    if cfg["type"] == "ds2":
        device, device_ids = prepare_device(cfg["n_gpu"])
        text_encoder = CTCCharTextEncoder()
        text_encoder.load_lm()
        arch = OmegaConf.load(cfg["model"])

        arch["n_class"] = len(text_encoder)

        model = instantiate(arch)
        model = model.to(device)
        if len(device_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=device_ids)

        checkpoint = torch.load(cfg["checkpoint_path"], map_location=device)
        state_dict = checkpoint
        model.load_state_dict(state_dict)
        model.eval()
    elif cfg["type"] == "whisper":
        m = cfg["model"]
        if m not in ["tiny", "base", "small", "medium", "large", "large-v1", "large-v2", "large-v3",
                     "tiny.en", "base.en", "small.en", "medium.en"]:
            raise KeyError
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = whisper.load_model(name=m, download_root=cfg["checkpoint_path"]).to(device)
        model.transcribe
    else:
        raise KeyError

    filepath = cfg["filepath"]
    output_dir = cfg["output_dir"]
    sr = cfg["sr"]

    if os.path.isfile(filepath):
        audio, filepath = load_n_process_audio(filepath, output_dir, sr)

        filename = filepath.split(".")[0].split("/")[-1]
        directory_save_file = os.path.join(output_dir, filename)

        os.makedirs(directory_save_file, exist_ok=True)

        def transcribe_audio_ds2(audio_segment: torch.Tensor):
            with torch.inference_mode():
                wave2spec = ta.transforms.MelSpectrogram()
                audio_segment_spec = wave2spec(audio_segment)
                audio_segment_spec = torch.log(audio_segment_spec + 1e-5)

                length = torch.Tensor([audio_segment_spec.shape[-1]]).type(torch.int)
                audio_segment_spec = audio_segment_spec.to(device)
                length = length.to(device)

                output = model(audio_segment_spec, length)

                log_probs = torch.log_softmax(output["logits"], dim=-1)
                log_probs_length = model.transform_input_lengths(length)
                #probs = log_probs.exp().cpu()
                #argmax = probs.argmax(-1)[:int(log_probs_length)]
                #result = text_encoder.ctc_decode_enhanced(argmax)
                #result = text_encoder.ctc_beam_search(log_probs.squeeze(), log_probs_length, beam_size=20)[0].text
                with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
                    result = text_encoder.ctc_beam_search_lm(log_probs, log_probs_length, beam_size=5000, pool=pool)[0]
                    print(result)
                    return result
                
        def transcribe_audio_whisper(audio_segment: torch.Tensor):
            audio = whisper.pad_or_trim(audio_segment.flatten()).to(device)
            mel = whisper.log_mel_spectrogram(audio)
            options = whisper.DecodingOptions(language="en", without_timestamps=False)
            result = model.decode(mel, options)
            print(result)
            return result


        speech_segments = read_and_process_file(cfg["boundaries"])
        transcriptions = []
        print("Transcriptions:")
        for start_time, end_time in speech_segments:
            start = max(int(start_time * sr), 0)
            end = min(int(end_time * sr), audio.shape[-1])
            audio_segment = audio[..., start:end]
            if cfg["type"] == "ds2":
                transcription = transcribe_audio_ds2(audio_segment)
            elif cfg["type"] == "whisper":
                transcription = transcribe_audio_whisper(audio_segment)
            transcriptions.append(transcription)
        output_file_path = os.path.join(directory_save_file, (filename + "_asr.csv"))
        create_output_file(speech_segments, transcriptions, output_file_path)

    return output_file_path


if __name__ == "__main__":
    cfg = {
            "type": "whisper",
            "model": "small.en",
            "sr": 16000,
            "filepath": "/home/comp/Рабочий стол/AutoDub/output/vad/1_mono_speech_resampled/1_mono_speech_resampled.wav",
            "boundaries": "/home/comp/Рабочий стол/AutoDub/output/vad/1_mono_speech_resampled/1_mono_speech_resampled_boundaries.txt",
            "output_dir": "/home/comp/Рабочий стол/AutoDub/output/asr2",
            "checkpoint_path": "/home/comp/Рабочий стол/AutoDub/checkpoints/asr/whisper"
            }
    inference_asr(cfg)