import editdistance
import torch
from torch import Tensor


def _handle_empty_target_text(predicted_text) -> float:
    if predicted_text:
        return 1
    return 0


def calc_cer(target_text, predicted_text) -> float:
    if not target_text:
        return _handle_empty_target_text(predicted_text)
    return editdistance.eval(target_text, predicted_text) / len(target_text)


def calc_wer(target_text, predicted_text) -> float:
    if not target_text:
        return _handle_empty_target_text(predicted_text)
    target_text_words = target_text.split(' ')
    return editdistance.eval(target_text_words, predicted_text.split(' ')) / len(target_text_words)


def decode_text(encoder_method, inference_type: str, log_probs: Tensor, log_probs_length: Tensor, **kwargs):
    # log_probs: (batch, length, n_class)
    # log_probs_length: (batch,)
    predicted_texts = []
    log_probs = log_probs.detach().cpu()
    log_probs_length = log_probs_length.cpu().numpy()
    if inference_type in ['ctc_decode_enhanced', 'ctc_decode']:
        predictions = torch.argmax(log_probs, dim=-1).numpy()
        for log_prob_vec, length in zip(predictions, log_probs_length):
            pred_text = encoder_method(log_prob_vec[:length])
            predicted_texts.append(pred_text)
    elif inference_type == 'ctc_beam_search':
        for log_probs_tensor, length in zip(log_probs, log_probs_length):
            pred_text = encoder_method(log_probs_tensor, length, **kwargs)[0].text
            predicted_texts.append(pred_text)
    else:
        assert False, 'Unreachable'
    return predicted_texts