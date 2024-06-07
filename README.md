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

# Credits
This repository is based on a heavily modified fork
of [hw template](https://github.com/WrathOfGrapes/asr_project_template) repository.

## Cascaded
[Repository](https://github.com/tsurumeso/vocal-remover)

## BandIt
[Repository](https://github.com/kwatcharasupat/bandit) 
[Paper](https://paperswithcode.com/paper/a-generalized-bandsplit-neural-network-for)

```
@ARTICLE{10342812,
  author={Watcharasupat, Karn N. and Wu, Chih-Wei and Ding, Yiwei and Orife, Iroro and Hipple, Aaron J. and Williams, Phillip A. and Kramer, Scott and Lerch, Alexander and Wolcott, William},
  journal={IEEE Open Journal of Signal Processing}, 
  title={A Generalized Bandsplit Neural Network for Cinematic Audio Source Separation}, 
  year={2024},
  volume={5},
  number={},
  pages={73-81},
  doi={10.1109/OJSP.2023.3339428}}
```

## Speechbrain
[Repository](https://github.com/speechbrain/speechbrain)


```
@misc{speechbrain,
  title={{SpeechBrain}: A General-Purpose Speech Toolkit},
  author={Mirco Ravanelli and Titouan Parcollet and Peter Plantinga and Aku Rouhe and Samuele Cornell and Loren Lugosch and Cem Subakan and Nauman Dawalatabad and Abdelwahab Heba and Jianyuan Zhong and Ju-Chieh Chou and Sung-Lin Yeh and Szu-Wei Fu and Chien-Feng Liao and Elena Rastorgueva and François Grondin and William Aris and Hwidong Na and Yan Gao and Renato De Mori and Yoshua Bengio},
  year={2021},
  eprint={2106.04624},
  archivePrefix={arXiv},
  primaryClass={eess.AS},
  note={arXiv:2106.04624}
}
```

## Coqui-ai TTS
[Repository](https://github.com/coqui-ai/TTS)

## DeepSpeech2
[Paper](https://arxiv.org/abs/1512.02595)

```
@misc{amodei2015deep,
      title={Deep Speech 2: End-to-End Speech Recognition in English and Mandarin}, 
      author={Dario Amodei and Rishita Anubhai and Eric Battenberg and Carl Case and Jared Casper and Bryan Catanzaro and Jingdong Chen and Mike Chrzanowski and Adam Coates and Greg Diamos and Erich Elsen and Jesse Engel and Linxi Fan and Christopher Fougner and Tony Han and Awni Hannun and Billy Jun and Patrick LeGresley and Libby Lin and Sharan Narang and Andrew Ng and Sherjil Ozair and Ryan Prenger and Jonathan Raiman and Sanjeev Satheesh and David Seetapun and Shubho Sengupta and Yi Wang and Zhiqian Wang and Chong Wang and Bo Xiao and Dani Yogatama and Jun Zhan and Zhenyao Zhu},
      year={2015},
      eprint={1512.02595},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Whisper
[Repository](https://github.com/openai/whisper) 
[Paper](https://arxiv.org/abs/2212.04356)

```
@misc{radford2022robust,
      title={Robust Speech Recognition via Large-Scale Weak Supervision}, 
      author={Alec Radford and Jong Wook Kim and Tao Xu and Greg Brockman and Christine McLeavey and Ilya Sutskever},
      year={2022},
      eprint={2212.04356},
      archivePrefix={arXiv},
      primaryClass={eess.AS}
}
```
