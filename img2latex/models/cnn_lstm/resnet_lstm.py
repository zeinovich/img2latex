import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50, resnet101

# [TODO] add device member
# [TODO] add image_size member
# [TODO] add model ABC


class ResnetLSTM(nn.Module):
    def __init__(
        self,
        resnet_depth: int = 34,
        encoder_out: int = 512,
        emb_dim: int = 64,
        hidden_dim: int = 256,
        dropout_prob: float = 0.2,
        vocab: dict[str, int] = None,
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

        self._encoder_out = encoder_out
        self._hidden_dim = hidden_dim
        self._vocab = vocab
        # output dim is set to vocab_size
        # will raise TypeError if not specified
        self._output_dim = len(self._vocab)
        self._emb_dim = emb_dim
        self._max_len = max_output_length
        self._device = device

        if resnet_depth == 18:
            self.encoder = resnet18()
        if resnet_depth == 34:
            self.encoder = resnet34()
        if resnet_depth == 50:
            self.encoder = resnet50()
        if resnet_depth == 101:
            self.encoder = resnet101()

        self.encoder.fc = nn.Sequential(
            nn.Linear(self.encoder.fc.in_features, self._encoder_out)
        )

        self.decoder = nn.LSTMCell(
            input_size=self._emb_dim + self._encoder_out,
            hidden_size=self._hidden_dim,
        )
        # initializes hidden state from encoder output
        # for initial out_t encoded_img is used
        self.init_hidden = nn.Linear(
            self._encoder_out, self._hidden_dim, bias=False
        )
        self.init_cell = nn.Linear(
            self._encoder_out, self._hidden_dim, bias=False
        )

        self.dropout = nn.Dropout(p=dropout_prob)
        self.embedding = nn.Embedding(self._output_dim, self._emb_dim)
        self.fc = nn.Linear(self._hidden_dim, self._encoder_out, bias=False)
        self.relu = nn.LeakyReLU()
        self.fc_out = nn.Linear(self._encoder_out, self._output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        encoded_img = self.encoder(x)

        # outputs of shape (B, MAX_LEN, VOCAB_SIZE (OUTPUT_DIM))
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

        hidden_t, cell_t = self.decoder(input_t, (hidden, cell))
        hidden_t, cell_t = self.dropout(hidden_t), self.dropout(cell_t)

        out_t = self.fc(hidden_t).tanh()
        out_t = self.dropout(out_t)
        out_t = self.relu(out_t)
        logit = self.fc_out(out_t)

        return hidden_t, cell_t, out_t, logit

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        encoded_img = self.encoder(x)

        # outputs of shape (B, MAX_LEN, VOCAB_SIZE (OUTPUT_DIM))
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
