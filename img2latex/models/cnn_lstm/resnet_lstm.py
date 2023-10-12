import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50, resnet101
import torchvision.transforms as T
import torch.functional as F

# [TODO] CHANGE TOKENS AND PUT THEM INTO MODEL
START_TOKEN = 0
PAD_TOKEN = 1
END_TOKEN = 2


# [TODO] Add vocabulary and reverse vocabulary
class ResnetLSTM(nn.Module):
    def __init__(
        self,
        resnet_depth: int = 18,
        encoder_out: int = 256,
        hidden_dim: int = 256,
        output_dim: int = 256,
        emb_dim: int = 80,
        # lstm_layers: int = 3,
        dropout_prob: float = 0.2,
        device: str = "cpu",
    ) -> None:
        super().__init__()

        assert resnet_depth in [
            18,
            34,
            50,
            100,
        ], "resnet_depth must be one of [18, 34, 50, 101]"

        self.encoder_out = encoder_out
        self.output_dim = output_dim
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        # self.lstm_layers = lstm_layers
        self.device = device

        if resnet_depth == 18:
            self.encoder = resnet18()
        if resnet_depth == 34:
            self.encoder = resnet34()
        if resnet_depth == 50:
            self.encoder = resnet50()
        if resnet_depth == 101:
            self.encoder = resnet101()

        self.encoder.fc = nn.Sequential(
            nn.Linear(self.encoder.fc.in_features, self.encoder_out)
        )

        self.decoder = nn.LSTMCell(
            input_size=self.emb_dim + self.encoder_out,
            hidden_size=self.hidden_dim,
        )

        self.dropout = nn.Dropout(p=dropout_prob)
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.fc = nn.Linear(self.hidden_dim, self.encoder_out, bias=False)
        self.init_hidden = nn.Linear(
            self.encoder_out, self.hidden_dim, bias=False
        )
        self.init_cell = nn.Linear(
            self.encoder_out, self.hidden_dim, bias=False
        )
        self.fc_out = nn.Linear(256, output_dim)

    def forward(self, x: torch.Tensor, max_len: int) -> torch.Tensor:
        batch_size = x.size(0)
        encoded_img = self.encoder(x)

        # outputs of shape (B, MAX_LEN, VOCAB_SIZE (OUTPUT_DIM))
        outputs = torch.zeros(batch_size, max_len, self.output_dim)
        # 1st input is always <START_SEQ>
        input_token = torch.ones(batch_size, 1).long() * START_TOKEN
        output = encoded_img
        hidden = self.init_hidden(encoded_img)
        cell = self.init_cell(encoded_img)

        for t in range(1, max_len):
            hidden, cell, output, logit = self.decode(
                hidden, cell, output, input_token
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

        logit = self.fc_out(out_t)

        return hidden_t, cell_t, out_t, logit
