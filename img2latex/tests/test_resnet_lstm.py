import pytest
import json
import torch

from ..models.cnn_lstm import ResnetLSTM


@pytest.fixture()
def token2id() -> dict[str, int]:
    with open("../data/interim/token2id.json", "r") as f:
        token2id_ = json.load(f)
    return token2id_


@pytest.fixture()
def resnet18_lstm(token2id: dict[str, int]):
    return ResnetLSTM(
        resnet_depth=18,
        encoder_out=512,
        hidden_dim=256,
        emb_dim=64,
        max_output_length=512,
        dropout_prob=0.2,
        vocab=token2id,
    )


@pytest.fixture()
def resnet34_lstm(token2id: dict[str, int]):
    return ResnetLSTM(
        resnet_depth=34,
        encoder_out=512,
        hidden_dim=256,
        emb_dim=64,
        max_output_length=512,
        dropout_prob=0.2,
        vocab=token2id,
    )


@pytest.fixture()
def resnet50_lstm(token2id: dict[str, int]):
    return ResnetLSTM(
        resnet_depth=50,
        encoder_out=512,
        hidden_dim=256,
        emb_dim=64,
        max_output_length=512,
        dropout_prob=0.2,
        vocab=token2id,
    )


@pytest.fixture()
def resnet101_lstm(token2id: dict[str, int]):
    return ResnetLSTM(
        resnet_depth=101,
        encoder_out=512,
        hidden_dim=256,
        emb_dim=64,
        max_output_length=512,
        dropout_prob=0.2,
        vocab=token2id,
    )


def test_resnet_configured(
    resnet18_lstm: ResnetLSTM,
    resnet34_lstm: ResnetLSTM,
    resnet50_lstm: ResnetLSTM,
    resnet101_lstm: ResnetLSTM,
):
    BATCH_SIZE = 3
    INPUT_RESOLUTION = 224

    input_img = torch.randn(BATCH_SIZE, 3, INPUT_RESOLUTION, INPUT_RESOLUTION)
    output18 = resnet18_lstm(input_img)
    output34 = resnet34_lstm(input_img)
    output50 = resnet50_lstm(input_img)
    output101 = resnet101_lstm(input_img)

    assert output18.size() == (
        BATCH_SIZE,
        resnet18_lstm.max_len + 2,
        resnet18_lstm.output_dim,
    )
    assert output34.size() == (
        BATCH_SIZE,
        resnet34_lstm.max_len + 2,
        resnet34_lstm.output_dim,
    )
    assert output50.size() == (
        BATCH_SIZE,
        resnet50_lstm.max_len + 2,
        resnet50_lstm.output_dim,
    )
    assert output101.size() == (
        BATCH_SIZE,
        resnet101_lstm.max_len + 2,
        resnet101_lstm.output_dim,
    )
