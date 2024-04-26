import os
import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts


def lazy_tts(csv_filepath, output_dir, checkpoint_path):
    print("Loading model...")
    config = XttsConfig()
    config.load_json("/path/to/xtts/config.json")
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=checkpoint_path, use_deepspeed=True)
    model.cuda()

    print("Computing speaker latents...")
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=["reference.wav"])

    print("Inference...")
    out = model.inference(
        "It took me quite a long time to develop a voice and now that I have it I am not going to be silent.",
        "ru",
        gpt_cond_latent,
        speaker_embedding,
        temperature=0.7, # Add custom parameters here
    )
    torchaudio.save("xtts.wav", torch.tensor(out["wav"]).unsqueeze(0), 24000)