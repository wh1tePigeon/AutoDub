import os
from typing import List, NamedTuple, Tuple, Dict
from collections import defaultdict
from pathlib import Path

import torch
import shutil
import gzip
import multiprocessing
from pyctcdecode import build_ctcdecoder
from speechbrain.utils.data_utils import download_file

from .char_text_encoder import CharTextEncoder
from source.utils.util import ROOT_PATH


LM_MODELS_DIRECTORY = ROOT_PATH / 'lm_models/'
LM_MODELS_DIRECTORY.mkdir(exist_ok=True)

MODEL_URL = 'https://www.openslr.org/resources/11/3-gram.pruned.1e-7.arpa.gz'
VOCAB_URL = 'http://www.openslr.org/resources/11/librispeech-vocab.txt'

MODEL_PATH = LM_MODELS_DIRECTORY / '3-gram.pruned.1e-7.arpa'
VOCAB_PATH = LM_MODELS_DIRECTORY / 'librispeech-vocab.txt'


class Hypothesis(NamedTuple):
    text: str
    prob: float


class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: List[str] = None):
        super().__init__(alphabet)
        vocab = [self.EMPTY_TOK] + list(self.alphabet)
        self.ind2char = dict(enumerate(vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}
        self.lm_model = None

    def _correct_sentence(self, text: str) -> str:
        # Remove double spaces
        text = text.strip()
        while '  ' in text:
            text = text.replace('  ', ' ')
        return text

    def ctc_decode_enhanced(self, inds: List[int]) -> str:
        return self._correct_sentence(self.ctc_decode(inds))

    def ctc_decode(self, inds: List[int]) -> str:
        letters = [self.ind2char[ind] for ind in inds]

        # Store parsed letters and last letter
        result = []
        last_letter = self.EMPTY_TOK

        for letter in letters:
            if letter == last_letter:
                # Skip the same letter
                continue
            # Update last_letter
            last_letter = letter
            if letter == self.EMPTY_TOK:
                # Skip EMPTY_TOK
                continue
            # This letter is not the same as previous and it is not EMPTY_TOK
            result.append(letter)
        return ''.join(result)

    @staticmethod
    def _get_best_prefixes(state: Dict[Tuple[str, str], float], beam_size: int) -> Dict[Tuple[str, str], float]:
        # Calculate the probability of each prefix
        prefix_total_prob = defaultdict(float)
        for (pref, last_char), pref_prob in state.items():
            prefix_total_prob[pref] += pref_prob
        # Take only the best prefixes
        return dict(sorted(prefix_total_prob.items(), key=lambda kv: kv[1], reverse=True)[:beam_size])

    @staticmethod
    def _truncate_state_to_best_prefixes(state: Dict[Tuple[str, str], float], best_prefixes: Dict[str, float]) -> Dict[Tuple[str, str], float]:
        return {(pref, last_char): pref_prob for (pref, last_char), pref_prob in state.items() if pref in best_prefixes}

    def _extend_and_merge(self, probs_for_time_t: torch.Tensor, state: Dict[Tuple[str, str], float]) -> Dict[Tuple[str, str], float]:
        new_state = defaultdict(float)
        # Iterate over next possible characters
        for next_char_ind, next_char_prob in enumerate(probs_for_time_t.tolist()):
            # Iterate over last prefixes
            for (pref, last_char), pref_prob in state.items():
                next_char = self.ind2char[next_char_ind]
                # Find new prefix
                if next_char == last_char or next_char == self.EMPTY_TOK:
                    new_pref = pref
                else:
                    new_pref = pref + next_char
                # Add probability to prefix
                new_state[(new_pref, next_char)] += pref_prob * next_char_prob
        return new_state

    def ctc_beam_search(self, log_probs: torch.Tensor, log_probs_length: int, beam_size: int) -> List[Hypothesis]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        assert len(log_probs.shape) == 2
        log_probs = log_probs[:log_probs_length]
        char_length, voc_size = log_probs.shape
        assert char_length == log_probs_length
        probs = torch.exp(log_probs)

        assert voc_size == len(self.ind2char)
        # (prefix, last_token) -> prob
        state: dict[tuple[str, str], float] = {('', self.EMPTY_TOK): 1.0}
        # prefix -> prob
        best_prefixes: dict[str, float] = {'': 1.0}
        for probs_for_time_t in probs:
            # Remove unlikely prefixes
            state = self._truncate_state_to_best_prefixes(state, best_prefixes)
            # Do 1 dynamical programming step
            state = self._extend_and_merge(probs_for_time_t, state)
            # Calculate the prefixes with highest probabilities
            best_prefixes = self._get_best_prefixes(state, beam_size)
        # Return hypothesis with their probabilities
        hypos = [Hypothesis(self._correct_sentence(prefix), prob) for prefix, prob in best_prefixes.items()]
        return sorted(hypos, key=lambda x: x.prob, reverse=True)

    def _download_lm(self):
        """
        Download model if not downloaded before
        """
        if not MODEL_PATH.exists():
            extract_path = LM_MODELS_DIRECTORY / '3-gram.pruned.1e-7.arpa.gz'
            # Download file
            download_file(MODEL_URL, extract_path)
            # Extract file
            with gzip.open(extract_path, 'rb') as f_in, open(MODEL_PATH, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
            os.remove(str(extract_path))
            # Convert to lowercase
            with open(MODEL_PATH) as f:
                content = f.read()
            # Remove ' and " characters
            with open(MODEL_PATH, 'w') as f:
                f.write(content.lower().replace('\'', '').replace('"', ''))
        download_file(VOCAB_URL, VOCAB_PATH)

    def load_lm(self):
        """
        Load language model if not loaded before
        """
        if self.lm_model is not None:
            return
        # Download it if not downloaded before
        self._download_lm()
        # Read the vocabulary
        with open(VOCAB_PATH) as f:
            # Remove any ' and " signs from words
            unigram_list = [t.lower().strip().replace('\'', '').replace('"', '') for t in f.read().strip().split("\n")]
        # Construct language model
        self.lm_model = build_ctcdecoder(
            [''] + self.alphabet,  # Add <unk>
            str(MODEL_PATH),
            unigram_list
        )

    def ctc_beam_search_lm(self, log_probs_batch: torch.Tensor, log_probs_lengths: torch.Tensor, beam_size: int, pool: multiprocessing.Pool) -> List[str]:
    #def ctc_beam_search_lm(self, log_probs_batch: torch.Tensor, log_probs_lengths: torch.Tensor, beam_size: int) -> List[str]:
        """
        Beam search with language model
        It is performed in parallel by batches using pool
        """
        # Load language model if not loaded before
        self.load_lm()
        assert len(log_probs_batch) == len(log_probs_lengths)
        # Truncate and convert to numpy arrays
        log_probs_batch = [log_probs[:length].cpu().numpy() for log_probs, length in zip(log_probs_batch, log_probs_lengths)]
        # Perform parallel beam search
        pred_lm = self.lm_model.decode_batch(pool, log_probs_batch, beam_width=beam_size)
        # Correct sentence (remove double spaces)
        return [self._correct_sentence(sentence) for sentence in pred_lm]