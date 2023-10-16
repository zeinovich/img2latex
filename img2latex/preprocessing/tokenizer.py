import torch
import numpy as np

import re
import warnings
from typing import Union


class LaTEXTokenizer:
    """
    Tokenizer for LaTEX exspressions. Vocabulary is generated from IM2LATEX-100k

    ---
    Parameters
    ---
    id2token: dict[str. int]
        Mapping from tokens to indices
    max_len: int
        Maximum length of tokenized output. Defaults to 512 (based on 99,9% quantile of length of formulas in IM2LATEX-100k)
    """

    def __init__(self, token2id: dict[str, int], max_len: int = 512) -> None:
        self._token2id = {k: int(v) for k, v in token2id.items()}
        self._id2token = {int(v): k for k, v in token2id.items()}
        self._max_len = max_len

    def tokenize(
        self,
        x: list[str],
        return_tensors: bool = True,
        pad: bool = True,
    ) -> Union[torch.Tensor, list]:  # separate dots
        """
        Tokenize list of sentences.

        -----
        Parameters
        -----

        x: list[str]
            Input list of sentences

        ---
        Returns
        ---
        torch.Tensor
            Tensor of indices with shape (B, MAX_LEN)"""

        x = [s.replace(".", " . ") for s in x]
        # separate digits
        x = [re.sub(r"(\d)", r" \1", s) for s in x]
        x = [s.strip().split() for s in x]
        x = [
            [self._token2id.get(token, self._token2id["<UNK>"]) for token in s]
            for s in x
        ]

        if any(self._token2id["<UNK>"] in s for s in x):
            warnings.warn(
                "Got unknown token. May affect final result",
            )

        # insert start and end tokens
        x = [[self._token2id["<SOS>"]] + s for s in x]
        x = [s + [self._token2id["<EOS>"]] for s in x]
        x = [s[: self.max_len] for s in x]

        if pad:
            # pad sequences to max length
            x = [
                s + [self._token2id["<PAD>"]] * (self.max_len - len(s))
                for s in x
            ]

        if return_tensors:
            x = torch.Tensor(x)

        return x

    def decode(
        self, x: Union[torch.Tensor, np.ndarray, list[int]]
    ) -> list[str]:
        """
        Decodes input array of indices into a string(s)

        ---
        Parameters
        ---
        x: Union[torch.Tensor, np.ndarray, list[int]]
            Input sequence of sequences of indices

        ---
        Returns
        ---
        list[str]
            List of decoded strings (ready to be displayed as LaTEX)
        """
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().tolist()
        if isinstance(x, np.ndarray):
            x = x.tolist()

        x = [
            [
                self._id2token[i]
                for i in s
                if i not in list(self.special_tokens.values())
                # ignore special tokens
            ]
            for s in x
        ]
        x = [" ".join(s) for s in x]
        x = [re.sub(r" +", " ", s) for s in x]
        x = [s.strip() for s in x]

        return x

    @property
    def special_tokens(self) -> dict[str, int]:
        """
        Get mapping (str2int) of special tokens
        """
        tokens = ["<SOS>", "<PAD>", "<EOS>"]
        return {k: self.token2id.get(k, None) for k in tokens}

    @property
    def max_len(self) -> int:
        """
        Get maximum length of sequence
        """
        return self._max_len

    @property
    def id2token(self) -> dict[int, str]:
        """
        Get vocabulary (int2str)
        """
        return self._id2token

    @property
    def token2id(self) -> dict[str, int]:
        """
        Get vocabulary (str2int)
        """
        return self._token2id
