import torch
import sentencepiece as spm

from .ctc_char_text_encoder import CTCCharTextEncoder


class CTCCharBpeEncoder(CTCCharTextEncoder):
    def __init__(self, model_prefix: str):
        super().__init__(None)
        self.sp_model = spm.SentencePieceProcessor(model_file=str(model_prefix) + '.model')
        assert self.sp_model.unk_id() == 0
        self.ind2char = {}
        self.alphabet = None
        self.char2ind = None
        for i in range(self.sp_model.vocab_size()):
            c = self.sp_model.IdToPiece(i).replace('▁', ' ')
            if '<' in c:
                c = self.EMPTY_TOK
            self.ind2char[i] = c

    def encode(self, text) -> torch.Tensor:
        text = self.normalize_text(text)
        return torch.Tensor(self.sp_model.Encode(text)).unsqueeze(0)
