# Overview
This project aims to use various deep learning models to perform automatic dubbing from english to russian language. The general pipeline looks like this:
<p align="center">
<img src="docs/pipe.png" alt="Overall pipeline"
width="800px"></p>

Baseline:
* Firstly, we separate video and audio
* Then we apply cascadedNet to separate voices from the background
* Then we apply speechbrain`s VAD model to detect segments boundaries
* After that, we apply DeepSpeech2 to segments we obtained from VAD
* After getting such pseudo-script we can translate it with googletrans
* The we use TTS and translated script to generate dubbed segments
* At the and, we concat video, background and new segments to get fully dubbed audio

Improvements:
* cascadedNet was replaced with BandIt
* DeepSpeech2 & VAD were replaced with whisper
* googletrans was replaced with LM
* After getting script, all speech segments can be diarized to improve TTS quality



# Installation guide
## Clone this repo:

```shell
git clone https://github.com/wh1tePigeon/AutoDub
```

## Install requirements:

```shell
pip install -r requirements.txt
```
## Download necessary checkpoints:

```shell
python download_checkpoints.py
```

# Repo structure:
```shell
├── README.md             <- Top-level README.
├── requirements.txt      <- project requirements.
├── dub_baseline.py       <- baseline dubbing code.
├── dub.py                <- Improved dubbing code.
├── download_checkpoints.py   <- script for downloading necessary checkpoints.
│
├── input                 <- input test examples.
├── docs                  <- main repo docs. 
│   
│
└── source                   <- main code directory.
    ├── augmentations            <- data augmintation functions
    ├── base                     <- base classes
    ├── configs                  <- necessary configs
    ├── datasets                 <- necessary datasets
    ├── inference                <- main inference functions
    ├── logger                   <- supportive logger
    ├── loss                     <- model`s losses
    ├── metric                   <- model`s metrics  
    ├── model                    <- model`s architectures
    ├── text_encoder             <- DS2 text encoders
    ├── trainer                  <- model`s train pipelines
    ├── train_model              <- initiate model`s train pipelines
    └── utils                    <- utils
```
#Credits

This repository is based on a heavily modified fork
of [hw template](https://github.com/WrathOfGrapes/asr_project_template) repository.


