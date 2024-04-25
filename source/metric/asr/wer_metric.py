from typing import List

from torch import Tensor

from source.base.base_metric import BaseMetric
from source.base.base_text_encoder import BaseTextEncoder
from source.metric.asr.utils import calc_wer, decode_text


class WERMetric(BaseMetric):
    def __init__(self, text_encoder: BaseTextEncoder, inference_type: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inference_type = inference_type
        self.encoder_method = getattr(text_encoder, inference_type)

    def __call__(self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs):
        predicted_texts = decode_text(self.encoder_method, self.inference_type, log_probs, log_probs_length, **self.kwargs)
        wers = []
        for pred_text, target_text in zip(predicted_texts, text):
            target_text = BaseTextEncoder.normalize_text(target_text)
            wers.append(calc_wer(target_text, pred_text))
        return sum(wers) / len(wers) * 100