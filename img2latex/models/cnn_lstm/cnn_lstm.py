import torch
import torch.nn as nn

from ..base_model import BaseIm2SeqModel

# [TODO] add model ABC


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
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x
            of dimensions (B, C, H, W)

        Returns
        -------
        torch.Tensor
            of dimensions (B, C, H, W)
        """
        c = self.conv(x)
        r = self.relu(c)
        return r


class CNNLSTM(BaseIm2SeqModel):
    def __init__(
        self,
        resnet_depth: int = 18,
        emb_dim: int = 24,
        hidden_dim: int = 512,
        dropout_prob: float = 0,
        vocab_len: int = None,
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
        self._img_size = 256

        # import weights from corresponding module
        self.encoder = nn.Sequential(  # img_size = 256
            ConvBlock(in_channels=1, out_channels=32),
            ConvBlock(in_channels=32, out_channels=32),
            nn.AvgPool2d(2, 2),  # img_size = 128
            ConvBlock(in_channels=32, out_channels=64),
            ConvBlock(in_channels=64, out_channels=64),
            nn.AvgPool2d(2, 2),  # img_size = 64
            ConvBlock(in_channels=64, out_channels=128),
            ConvBlock(in_channels=128, out_channels=128),
            nn.AvgPool2d(2, 2),  # img_size = 32
            ConvBlock(in_channels=128, out_channels=256),
            ConvBlock(in_channels=256, out_channels=256),
            nn.AvgPool2d(2, 2),  # img_size = 16
            ConvBlock(in_channels=256, out_channels=256),
            nn.MaxPool2d(2, 2),  # img_size = 8
            # (B, 256, 8, 8)
            nn.Flatten(),
            nn.Linear(in_features=256 * 8 * 8, out_features=self._encoder_out),
        )

        self.decoder = nn.LSTM(
            input_size=self._emb_dim + self._encoder_out,
            hidden_size=self._hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        self.dropout = nn.Dropout(p=dropout_prob)
        self.embedding = nn.Embedding(
            self._output_dim, self._emb_dim, padding_idx=self._vocab["<PAD>"]
        )
        self.fc_out = nn.Sequential(
            nn.Linear(self._hidden_dim, self._hidden_dim),
            nn.Linear(self._hidden_dim, self._output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        encoded_img = self.encoder(x)

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

        hidden = encoded_img
        cell = torch.zeros(batch_size, self._hidden_dim)
        output = torch.zeros(batch_size, self._hidden_dim)

        for t in range(1, self._max_len):
            hidden, cell, output, logit = self.decode(
                hidden=hidden,
                cell=cell,
                out_t=output,
                input_token=input_token,
            )

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
        logit = self.fc_out(out_t)

        return hidden_t, cell_t, out_t, logit

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

    @property
    def img_size(self):
        return self._img_size
