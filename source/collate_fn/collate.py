import logging
from typing import List

import torch

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """

    result_batch = {}

    for name in ('audio', 'text_encoded', 'spectrogram'):
        len_name = f'{name}_length'
        result_batch[len_name] = [dt[name].size(-1) for dt in dataset_items]
        if name == 'spectrogram':
            batch = torch.zeros(len(result_batch[len_name]),
                                dataset_items[0]['spectrogram'].shape[1],
                                max(result_batch[len_name]))
        else:
            batch = torch.zeros(len(result_batch[len_name]), max(result_batch[len_name]))
        for i in range(len(result_batch[len_name])):
            if name == 'spectrogram':
                batch[i, :, :result_batch[len_name][i]] = dataset_items[i][name]
            else:
                batch[i, :result_batch[len_name][i]] = dataset_items[i][name]
        result_batch[name] = batch
        result_batch[len_name] = torch.tensor(result_batch[len_name]).long()

    for name in ('duration', 'text', 'audio_path'):
        result_batch[name] = [dt[name] for dt in dataset_items]

    return result_batch