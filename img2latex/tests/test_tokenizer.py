import pytest
import json
import warnings

from ..preprocessing.tokenizer import LaTEXTokenizer


@pytest.fixture()
def tokenizer():
    with open("./data/processed/token2id.json", "r") as f:
        token2id = json.load(f)

    return LaTEXTokenizer(token2id)


def test_encode_decode(tokenizer: LaTEXTokenizer):
    TEST_INPUT = [
        "g \\approx 3 - \\sqrt 3 - 0 . 9 1 7 7 f _ { 0 } ^ { 2 } \\; .",
        "g \\approx 3 - \\sqrt 3 - 0 . 9 1 7 7 f _ { 0 } ^ { 2 } \\; .",
    ]
    max_len = 512
    encoded_input = tokenizer.tokenize(TEST_INPUT, max_len=max_len)
    decoded_output = tokenizer.decode(encoded_input)
    assert decoded_output == TEST_INPUT
    assert encoded_input.size(1) == max_len


def test_encode_decode_numpy(tokenizer: LaTEXTokenizer):
    TEST_INPUT = [
        "g \\approx 3 - \\sqrt 3 - 0 . 9 1 7 7 f _ { 0 } ^ { 2 } \\; .",
        "g \\approx 3 - \\sqrt 3 - 0 . 9 1 7 7 f _ { 0 } ^ { 2 } \\; .",
    ]
    max_len = 512
    encoded_input = tokenizer.tokenize(TEST_INPUT, max_len=max_len)
    encoded_input = encoded_input.detach().cpu().numpy()
    decoded_output = tokenizer.decode(encoded_input)
    assert decoded_output == TEST_INPUT
    assert encoded_input.shape[1] == max_len


def test_encode_decode_unknown_token(tokenizer: LaTEXTokenizer):
    # \\eriuert doesn't exist in LaTEX vocab
    TEST_INPUT = [
        "g \\eriuert 3 - \\sqrt 3 - 0 . 9 1 7 7 f _ { 0 } ^ { 2 } \\; ."
    ]

    TEST_OUTPUT = [
        "g <UNK> 3 - \\sqrt 3 - 0 . 9 1 7 7 f _ { 0 } ^ { 2 } \\; ."
    ]
    max_len = 512
    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")
        # Trigger a warning.
        encoded_input = tokenizer.tokenize(TEST_INPUT, max_len=max_len)
        assert "Got unknown token" in str(w[-1].message)

    decoded_output = tokenizer.decode(encoded_input)

    assert decoded_output == TEST_OUTPUT
    assert encoded_input.size(1) == max_len


def test_get_special_tokens(tokenizer: LaTEXTokenizer):
    assert isinstance(tokenizer.special_tokens, dict)
    assert isinstance(tokenizer.special_tokens.get("<EOS>"), int)


def test_vocabs(tokenizer: LaTEXTokenizer):
    assert len(tokenizer.id2token) == len(tokenizer.token2id)
