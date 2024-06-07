# Inference
## source.inference.asr.inference_asr:

*def inference_asr(cfg)*

&ensp; **parameters**:

&ensp; &ensp; *cfg*

&ensp; **return**:

&ensp; &ensp; *output_file_path* - path to csv file containing timestamps and transcribed speech



## source.inference.asr.inference_asr_wtime:

*def inference_asr_wtime(cfg)*

&ensp; **parameters**:

&ensp; &ensp; *cfg*

&ensp; **return**:

&ensp; &ensp; *filepath* - path to the audio

&ensp; &ensp; *output_file_path* - path to csv file containing timestamps and transcribed speech



## source.inference.bsrnn.inference_bsrnn:

*def inference_bsrnn(cfg)*

&ensp; **parameters**:

&ensp; &ensp; *cfg*

&ensp; **return**:

&ensp; &ensp; *speech_save_path* - path to speech audiofile 

&ensp; &ensp; *background_save_path* - path to background audiofile


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


## source.inference.translate.inference_translate:

*def translate_file(cfg)*

&ensp; **parameters**:

&ensp; &ensp; *cfg* - 


&ensp; **return**:

&ensp; &ensp; *output*  - path to csvfile with translation


## source.inference.tts.inference_tts:

*def tts(csv_filepath, output_dir, filename, target_sr, checkpoint_path)*

&ensp; **parameters**:

&ensp; &ensp; *csv_filepath* - path to csvfile with translated text and labeles

&ensp; &ensp; *output_dir* - path to directory for saving

&ensp; &ensp; *filename* - name for saving resulting csv

&ensp; &ensp; *target_sr* - target sampling rate

&ensp; &ensp; *checkpoint_path* - path to TTS`s model checkpoint

&ensp; **return**:

&ensp; &ensp; *new_csv_path* - path to csv file containing paths to generated speech segments


## source.inference.tts.inference_tts_lazy:

*def lazy_tts(csv_filepath, output_dir, filename, target_sr, checkpoint_path)*

&ensp; **parameters**:

&ensp; &ensp; *csv_filepath* - path to csvfile with translated text

&ensp; &ensp; *output_dir* - path to directory for saving

&ensp; &ensp; *filename* - name for saving resulting csv

&ensp; &ensp; *target_sr* - target sampling rate

&ensp; &ensp; *checkpoint_path* - path to TTS`s model checkpoint

&ensp; **return**:

&ensp; &ensp; *new_csv_path* - path to csv file containing paths to generated speech segments



## source.inference.vad.inference_vad:

*def inference_vad(cfg)*

&ensp; **parameters**:

&ensp; &ensp; *cfg* - 


&ensp; **return**:

&ensp; &ensp; *filepath* - path to audiofile

&ensp; &ensp; *path_to_log_file* - path to txt file containing boundaries
