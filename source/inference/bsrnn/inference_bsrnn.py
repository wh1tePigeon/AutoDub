import os
import sys
import torch
import torchaudio as ta
from hydra.utils import instantiate
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from source.utils.util import prepare_device
from source.utils.process_audio import load_n_process_audio
from source.utils.fader import OverlapAddFader
from omegaconf import OmegaConf
from tqdm import tqdm


def inference_bsrnn(cfg):
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

    filepath = cfg["filepath"]
    output_dir = cfg["output_dir"]
    sr = cfg["sr"]

    if os.path.isfile(filepath):
        audio, filepath = load_n_process_audio(filepath, output_dir, sr)
  
        # move audio to gpu
        audio = audio.to(device)

        with torch.inference_mode():
            def forward(audio):
                _, output = model({"audio": {"mixture": audio},})
                return output["audio"]
            
            if audio.shape[-1] / sr > cfg["max_len"]:
                if cfg["use_fader"]:
                    audio = audio.unsqueeze(0)
                    fader = OverlapAddFader(window_type=cfg["window_type"],
                                            chunk_size_second=cfg["chunk_size_second"],
                                            hop_size_second=cfg["hop_size_second"],
                                            fs=sr,
                                            batch_size=cfg["batch_size"])
                    
                    output = fader(audio, lambda a: forward(a))

                    speech = output["speech"]
                    speech = speech.reshape(1, -1)
                    speech = speech.to("cpu")

                
                else:
                    speech_segments = []
                    segment_len = int(sr * cfg["max_len"])
                    amount_of_segments = int(audio.shape[-1] // segment_len)
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
                
            else:
                audio = audio.unsqueeze(0)
                output = forward(audio)
            
                speech = output["speech"]
                speech = speech.reshape(1, -1)
                speech = speech.to("cpu")
            
            speech = speech.reshape(1, -1)
            audio = audio.reshape(1, -1)
            background = audio - speech

            filename = filepath.split(".")[0].split("/")[-1]

            directory_save_file = os.path.join(output_dir, filename)
            os.makedirs(directory_save_file, exist_ok=True)
            
            speech_save_path = os.path.join(directory_save_file, (filename + "_speech.wav"))
            background_save_path = os.path.join(directory_save_file, (filename + "_background.wav"))

            speech = speech.to("cpu")
            background = background.to("cpu")

            ta.save(speech_save_path, speech, sample_rate=sr)
            ta.save(background_save_path, background, sample_rate=sr)

            return [speech_save_path, background_save_path]


if __name__ == "__main__":
    cfg = {
        "n_gpu": 1,
        "model": "/home/comp/Рабочий стол/AutoDub/source/configs/bsrnn/arch/model_conf.yaml",
        "sr": 44100,
        "filepath" : "/home/comp/Рабочий стол/AutoDub/input/3.wav",
        "output_dir": "/home/comp/Рабочий стол/AutoDub/output/bsrnn",
        "checkpoint_path": "/home/comp/Рабочий стол/AutoDub/checkpoints/bsrnn/main.pth",
        "window_type": "hann",
        "chunk_size_second": 6.0,
        "hop_size_second": 0.5,
        "batch_size": 4,
        "use_fader" : False,
        "max_len" : 5.0
    }

    inference_bsrnn(cfg)