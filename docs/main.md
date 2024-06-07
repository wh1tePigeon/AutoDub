# Inference
## source.inference.asr.inference_asr:

*def inference_asr(cfg)*

&ensp; **parameters**:

&ensp; &ensp; *cfg*

&ensp; **return**:

&ensp; &ensp; *output_file_path* - Path to csv file containing timestamps and transcribed speech



## source.inference.asr.inference_asr_wtime:

*def inference_asr_wtime(cfg)*

&ensp; **parameters**:

&ensp; &ensp; *cfg*

&ensp; **return**:

&ensp; &ensp; *filepath* - Path to the audio

&ensp; &ensp; *output_file_path* - Path to csv file containing timestamps and transcribed speech



## source.inference.bsrnn.inference_bsrnn:

*def inference_bsrnn(cfg)*

&ensp; **parameters**:

&ensp; &ensp; *cfg*

&ensp; **return**:

&ensp; &ensp; *speech_save_path* - Path to speech audiofile 

&ensp; &ensp; *background_save_path* - Path to background audiofile


## source.inference.cascaded.inference_cascaded:

*def inference_cascaded(cfg)*

&ensp; **parameters**:

&ensp; &ensp; *cfg*

&ensp; **return**:

&ensp; &ensp; *vocal_save_path* **or** *vocal* - either path to speech audiofile or torch.Tensor of speech

&ensp; &ensp; *background_save_path* **or** *background* - either path to background audiofile or torch.Tensor of background

&ensp; &ensp; *sr* - sampling rate


## source.inference.diarize.inference_diarization:

*def label_speakers(audio_filepath, csv_filepath, output_dir, cluster_type, cluster_cfg)*

&ensp; **parameters**:

&ensp; &ensp; *audio_filepath* - path to audiofile with speech

&ensp; &ensp; *csv_filepath* - path to csvfile with timestamps

&ensp; &ensp; *output_dir* - path to directory for saving resulting csv

&ensp; &ensp; *cluster_type* - clustering algorithm

&ensp; &ensp; *cluster_cfg* - clustering algorithm`s config

&ensp; **return**:

&ensp; &ensp; *new_csv_path*  - path to csvfile with labels
