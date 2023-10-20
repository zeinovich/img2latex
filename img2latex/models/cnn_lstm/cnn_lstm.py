import torch
import torch.nn as nn

from ..base_model import BaseIm2SeqModel


class ConvBlock(nn.Module):
    """
    Simple 3x3 conv with padding size 1 (to leave the input size unchanged), followed by a ReLU.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.relu = nn.LeakyReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = self.conv(x)
        r = self.relu(c)
        return r


class CNNLSTM(BaseIm2SeqModel):
    def __init__(
        self,
        emb_dim: int = 80,
        hidden_dim: int = 512,
        dropout_prob: float = 0,
        vocab_len: int = None,
        max_output_length: int = 512,
        device: str = "cpu",
    ) -> None:
        super().__init__()

        # At least special tokens must be present
        assert vocab_len > 4, "Vocabulary length must be at least 4"

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

        self.encoder = nn.Sequential(
            ConvBlock(in_channels=1, out_channels=64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(in_channels=64, out_channels=128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(in_channels=128, out_channels=256),
            ConvBlock(in_channels=256, out_channels=256),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            ConvBlock(
                in_channels=256, out_channels=self._encoder_out, padding=0
            ),
        )

        self.decoder = nn.LSTM(
            input_size=self._emb_dim + self._encoder_out,
            hidden_size=self._hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        self.dropout = nn.Dropout(p=dropout_prob)
        self.embedding = nn.Embedding(
            self._output_dim,
            self._emb_dim,  # padding_idx=self._vocab["<PAD>"]
        )
        self.hidden0_fc = nn.Sequential(
            nn.Linear(self._encoder_out, self._hidden_dim)
        )
        self.fc_out = nn.Sequential(
            nn.Linear(self._hidden_dim, self._output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)

        # INITIALIZE OUTPUT TENSORS
        # outputs of shape (B, MAX_LEN, VOCAB_SIZE (OUTPUT_DIM))
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

        # ENCODE IMAGE
        encoded_img = self.encoder(x)
        encoded_img = encoded_img.permute(
            0, 2, 3, 1
        )  # make B * H * W * HIDDEN_DIM to use contiguous

        _, H, W, _ = encoded_img.size()

        encoded_img = encoded_img.contiguous().view(
            batch_size,
            H * W,
            self._encoder_out,
        )  # [B, HIDDEN_DIM, H * W]
        encoded_img = encoded_img.mean(dim=1).to(
            self._device
        )  # [B, HIDDEN_DIM]

        hidden = self.hidden0_fc(encoded_img)
        cell = torch.zeros(batch_size, self._hidden_dim).to(self._device)
        output = torch.zeros(batch_size, self._hidden_dim).to(self._device)

        for t in range(1, self._max_len):
            hidden, cell, output = self.decode(
                hidden=hidden,
                cell=cell,
                out_t=output,
                input_token=input_token,
            )

            logit = self.fc_out(output)
            outputs[:, t, ...] = logit
            input_token = torch.argmax(logit, 1)

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

        return hidden_t, cell_t, out_t

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        encoded_img = self.encoder(x)

        hidden = encoded_img
        cell = torch.zeros(batch_size, self._hidden_dim)
        output = torch.zeros(batch_size, self._hidden_dim)

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
