import logging
from typing import List
import torch
from torch.nn.utils.rnn import pad_sequence


logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """
    try:
        # Group values by keys
        batch = {key: [dataset_items[i][key] for i in range(len(dataset_items))] for key in dataset_items[0].keys()}
        # Stack duration
        batch['duration'] = torch.Tensor(batch['duration'])
        for key in ['audio', 'spectrogram', 'text_encoded']:
            # Find length
            batch[f'{key}_length'] = torch.Tensor([t.size(-1) for t in batch[key]]).long()
            # Stack padded fields
            batch[key] = pad_sequence([t.squeeze().movedim(-1, 0) for t in batch[key]], batch_first=True).movedim(1, -1)
    except IndexError:
        print(f'ERROR in collate_fn')
        return {}
    return batch