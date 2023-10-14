import pytest
import json

from ..models.tokenizer import LaTEXTokenizer


@pytest.fixture()
def tokenizer():
    with open("../data/interim/token2id.json", "r") as f:
        token2id = json.load(f)

    return LaTEXTokenizer(token2id)


def test_encode_decode(tokenizer: LaTEXTokenizer):
    TEST_INPUT = [
        "g \\approx 3 - \\sqrt 3 - 0 . 9 1 7 7 f _ { 0 } ^ { 2 } \\; .",
        "g \\approx 3 - \\sqrt 3 - 0 . 9 1 7 7 f _ { 0 } ^ { 2 } \\; .",
    ]

    encoded_input = tokenizer.tokenize(TEST_INPUT)
    decoded_output = tokenizer.decode(encoded_input)
    assert decoded_output == TEST_INPUT
    assert encoded_input.size(1) == tokenizer.max_len


def test_encode_decode_numpy(tokenizer: LaTEXTokenizer):
    TEST_INPUT = [
        "g \\approx 3 - \\sqrt 3 - 0 . 9 1 7 7 f _ { 0 } ^ { 2 } \\; .",
        "g \\approx 3 - \\sqrt 3 - 0 . 9 1 7 7 f _ { 0 } ^ { 2 } \\; .",
    ]

    encoded_input = tokenizer.tokenize(TEST_INPUT)
    encoded_input = encoded_input.detach().cpu().numpy()
    decoded_output = tokenizer.decode(encoded_input)
    assert decoded_output == TEST_INPUT
    assert encoded_input.shape[1] == tokenizer.max_len


def test_encode_decode_unknown_token(tokenizer: LaTEXTokenizer):
    TEST_INPUT = [
        "g \\eriuert 3 - \\sqrt 3 - 0 . 9 1 7 7 f _ { 0 } ^ { 2 } \\; ."
    ]

    TEST_OUTPUT = [
        "g <UNK> 3 - \\sqrt 3 - 0 . 9 1 7 7 f _ { 0 } ^ { 2 } \\; ."
    ]

    encoded_input = tokenizer.tokenize(TEST_INPUT)
    decoded_output = tokenizer.decode(encoded_input)
    assert decoded_output == TEST_OUTPUT
    assert encoded_input.size(1) == tokenizer.max_len


def test_get_special_tokens(tokenizer: LaTEXTokenizer):
    assert isinstance(tokenizer.special_tokens, dict)
    assert isinstance(tokenizer.special_tokens.get("<EOS>"), int)
