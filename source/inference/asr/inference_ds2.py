import os
import multiprocessing
from pathlib import Path
import sys
import hydra
from hydra.utils import instantiate
import torch
import torchaudio as ta
from omegaconf import DictConfig
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from source.utils.util import prepare_device, CONFIGS_PATH, CHECKPOINTS_DEFAULT_PATH, OUTPUT_DEFAULT_PATH
from source.utils.process_input_audio import load_n_process_audio
from source.text_encoder.ctc_char_text_encoder import CTCCharTextEncoder
import csv
from omegaconf import OmegaConf


#CONFIG_ASR_PATH = CONFIGS_PATH / 'asr'
#CONFIG_ASR_NAME = "main"
#ASR_CHECKPOINT_PATH = CHECKPOINTS_DEFAULT_PATH / 'asr' / 'main.pth'
#ASR_OUTPUT_PATH = OUTPUT_DEFAULT_PATH / 'asr'
#REQUIRED_SR = 16000


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


#@hydra.main(config_path=str(CONFIG_ASR_PATH), config_name="inference")
def inference_asr(cfg):
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

    filepath = cfg["filepath"]
    output_dir = cfg["output_dir"]
    sr = cfg["sr"]

    if os.path.isfile(filepath):
        audio, filepath = load_n_process_audio(filepath, output_dir, sr)

        filename = filepath.split(".")[0].split("/")[-1]
        directory_save_file = os.path.join(output_dir, filename)

        #if not os.path.exists(directory_save_file):
        #    os.mkdir(directory_save_file)
        os.makedirs(directory_save_file, exist_ok=True)

        def transcribe_audio(audio_segment: torch.Tensor):
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
                    print("-------------")
                    return result


        speech_segments = read_and_process_file(cfg["boundaries"])
        transcriptions = []
        print("Transcriptions:")
        for start_time, end_time in speech_segments:
            start = max(int(start_time * sr), 0)
            end = min(int(end_time * sr), audio.shape[-1])
            audio_segment = audio[..., start:end]
            transcription = transcribe_audio(audio_segment)
            transcriptions.append(transcription)
        output_file_path = os.path.join(directory_save_file, (filename + "_asr.csv"))
        create_output_file(speech_segments, transcriptions, output_file_path)

    return output_file_path


if __name__ == "__main__":
    inference_asr()