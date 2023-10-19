import torch
import torch.nn as nn
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
)

from ..base_model import BaseIm2SeqModel
import sys

# [TODO] add device member
# [TODO] add image_size member
# [TODO] add model ABC


class ResnetLSTM(BaseIm2SeqModel):
    def __init__(
        self,
        resnet_depth: int = 18,
        emb_dim: int = 16,
        hidden_dim: int = 128,
        dropout_prob: float = 0,
        vocab_len: dict[str, int] = None,
        max_output_length: int = 512,
        device: str = "cpu",
    ) -> None:
        super().__init__()

        assert resnet_depth in [
            18,
            34,
            50,
            101,
        ], "resnet_depth must be one of [18, 34, 50, 101]"

        self._encoder_out = hidden_dim
        self._hidden_dim = hidden_dim
        self._vocab_len = vocab_len

        # standard vocabulary with special tokens
        self._vocab = {
            "<UNK>": 0,
            "<SOS>": 1,
            "<PAD>": 2,
            "<EOS>": 3,
        }
        # output dim is set to vocab_len
        self._output_dim = self._vocab_len
        self._emb_dim = emb_dim
        self._max_len = max_output_length
        self._device = device

        # import weights from corresponding module
        weights = getattr(
            sys.modules[__name__], f"ResNet{resnet_depth}_Weights"
        ).DEFAULT

        # load model and weights from torch.hub
        self.encoder = torch.hub.load(
            "pytorch/vision:v0.10.0",
            f"resnet{resnet_depth}",
            weights=weights,
        )

        for param in self.encoder.parameters():
            param.requires_grad = False

        self.encoder.fc = nn.Sequential(
            nn.Linear(self.encoder.fc.in_features, self._encoder_out)
        )

        self.decoder = nn.LSTM(
            input_size=self._emb_dim + self._encoder_out,
            hidden_size=self._hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.init_hidden = nn.Linear(self._encoder_out, self._hidden_dim)
        self.init_cell = nn.Linear(self._encoder_out, self._hidden_dim)
        self.init_output = nn.Linear(self._encoder_out, self._hidden_dim)

        self.dropout = nn.Dropout(p=dropout_prob)
        self.embedding = nn.Embedding(
            self._output_dim, self._emb_dim, padding_idx=self._vocab["<PAD>"]
        )
        self.fc_out = nn.Linear(self._hidden_dim, self._output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        encoded_img = self.encoder(x)

        # outputs of shape (B, MAX_LEN, VOCAB_len_SIZE (OUTPUT_DIM))
        outputs = (
            torch.ones(
                batch_size, self._max_len, self._output_dim, requires_grad=True
            )
            .type_as(x)
            .to(self._device)
            * self._vocab["<PAD>"]
        )
        # 1st input is always <START_SEQ>
        input_token = (
            torch.ones(batch_size, 1).type_as(x).to(self._device).long()
            * self._vocab["<SOS>"]
        )

        hidden = self.init_hidden(encoded_img)
        cell = self.init_cell(encoded_img)
        output = self.init_output(encoded_img)

        for t in range(1, self._max_len):
            hidden, cell, output, logit = self.decode(
                hidden=hidden,
                cell=cell,
                out_t=output,
                input_token=input_token,
            )

            outputs[:, t, ...] = logit
            input_token = torch.argmax(logit, 1)

            if input_token.item() == self._vocab["<EOS>"]:
                break

        return outputs

    def decode(
        self,
        hidden: tuple[torch.Tensor],
        cell: torch.Tensor,
        out_t: torch.Tensor,
        input_token: torch.Tensor,
    ) -> tuple[torch.Tensor]:
        prev_y = self.embedding(input_token).squeeze(1)
        input_t = torch.cat([prev_y, out_t], 1)
        out_t, (hidden_t, cell_t) = self.decoder(input_t, (hidden, cell))
        logit = self.fc_out(out_t)

        return hidden_t, cell_t, out_t, logit

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        encoded_img = self.encoder(x)

        # outputs of shape (B, MAX_LEN, VOCAB_LEN (OUTPUT_DIM))
        outputs = (
            torch.ones(batch_size, self._max_len, self._output_dim)
            .type_as(x)
            .long()
            .to(self._device)
            * self._vocab["<PAD>"]
        )
        # 1st input is always <START_SEQ>
        input_token = (
            torch.ones(batch_size, 1).type_as(x).to(self._device).long()
            * self._vocab["<SOS>"]
        )
        output = encoded_img
        hidden = self.init_hidden(encoded_img)
        cell = self.init_cell(encoded_img)

        for t in range(1, self._max_len):
            hidden, cell, output, logit = self.decode(
                hidden=hidden,
                cell=cell,
                out_t=output,
                input_token=input_token,
            )

            outputs[:, t, ...] = logit
            input_token = torch.argmax(logit, 1)

            if input_token == self._vocab["<EOS>"]:
                break

        return outputs

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    @property
    def output_dim(self):
        return self._output_dim

    @property
    def max_len(self):
        return self._max_len

    @property
    def device(self):
        return self._device
