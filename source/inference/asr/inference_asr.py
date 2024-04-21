import os
from pathlib import Path
import sys
import hydra
from hydra.utils import instantiate
import torch
import torchaudio as ta
from omegaconf import DictConfig
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from source.utils.util import prepare_device, CONFIGS_PATH, CHECKPOINTS_DEFAULT_PATH, OUTPUT_DEFAULT_PATH
from source.text_encoder.ctc_char_text_encoder import CTCCharTextEncoder

CONFIG_ASR_PATH = CONFIGS_PATH / 'asr'
CONFIG_ASR_NAME = "main"
ASR_CHECKPOINT_PATH = CHECKPOINTS_DEFAULT_PATH / 'asr' / 'main.pth'
ASR_OUTPUT_PATH = OUTPUT_DEFAULT_PATH / 'asr'
REQUIRED_SR = 16000


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


def create_output_file(speech_segments, transcriptions, output_file_path):
    with open(output_file_path, 'w') as file:
        for segment, transcription in zip(speech_segments, transcriptions):
            file.write(f"{segment[0]} {segment[1]} {transcription}\n")


@hydra.main(config_path=str(CONFIG_ASR_PATH), config_name="inference")
def inference_asr(cfg: DictConfig):
    for p in cfg["paths"]:
        assert os.path.exists(cfg["paths"][p])

    device, device_ids = prepare_device(cfg["n_gpu"])
    text_encoder = CTCCharTextEncoder()
    cfg["arch"]["n_class"] = len(text_encoder)

    model = instantiate(cfg["arch"])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    checkpoint = torch.load(cfg["paths"]["checkpoint"], map_location=device)
    state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    filepath = cfg["paths"]["input"]
    directory_save = cfg["paths"]["output"]
    if os.path.isfile(filepath):
        filename = os.path.splitext(filepath)[0]
        directory_save_file = os.path.join(directory_save, filename)

        if not os.path.exists(directory_save_file):
            os.mkdir(directory_save_file)
        audio, fs = ta.load(filepath)

        resampled = False

        # resample input
        if fs != REQUIRED_SR:
            print("Wrong samplerate! Resample to 16 kHz")
            audio = ta.functional.resample(
                audio,
                fs,
                REQUIRED_SR
            )
            resampled = True

        # save resampled audio
        if resampled:
            filepath = os.path.join(directory_save_file, (filename + "-resampled.wav"))
            ta.save(filepath, audio, REQUIRED_SR)
            print("Resampled audio saved in ", filepath)

        def transcribe_audio(audio_segment: torch.Tensor):
            with torch.inference_mode():
                audio_segment_spec = ta.transforms.MelSpectrogram()(audio_segment)
                length = torch.Tensor([audio_segment_spec.shape[-1]]).type(torch.int)
                audio_segment_spec = audio_segment_spec.to(device)
                length = length.to(device)

                output = model(audio_segment_spec, length)

                log_probs = torch.log_softmax(output["logits"], dim=-1)
                log_probs_length = model.transform_input_lengths(length)
                probs = log_probs.exp().cpu()
                argmax = probs.argmax(-1)[:int(log_probs_length)]
                #result = text_encoder.ctc_decode_enhanced(argmax)
                result = text_encoder.ctc_beam_search(log_probs.squeeze(), log_probs_length, beam_size=20)
                print(result)
                print("-------------")
        

        speech_segments = read_and_process_file(cfg["paths"]["boundaries"])
        transcriptions = []
        for start_time, end_time in speech_segments:
            start = max(int(start_time * REQUIRED_SR), 0)
            end = min(int(end_time * REQUIRED_SR), audio.shape[-1])
            audio_segment = audio[..., start:end]
            transcription = transcribe_audio(audio_segment)
            transcriptions.append(transcription)
        create_output_file(speech_segments, transcriptions, cfg["paths"]["output"])


if __name__ == "__main__":
    inference_asr()