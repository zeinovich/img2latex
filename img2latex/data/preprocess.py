import argparse
import json
import pandas as pd

from .utils import read_input, save_output
from ..preprocessing import LaTEXTokenizer


def cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Makes LaTEX vocabulary from a file. Outputs JSON file"
    )
    parser.add_argument(
        "input-file",
        type=str,
        help="Input file. Must be one of [csv, parquet, pickle]",
    )
    parser.add_argument(
        "output-file",
        type=str,
        help="Output file with tokenized and encoded LaTEX formulas. Must be one of [csv, parquet, pickle]",
    )
    parser.add_argument(
        "--vocab",
        type=str,
        help="JSON file with vocabulary (token2id)",
    )
    parser.add_argument(
        "--col-name",
        type=str,
        default=None,
        help="Column of DataFrame to be used. If set to None, DataFrame is required to have one column",
    )

    args = parser.parse_args()
    return args


def read_vocab(path: str) -> dict[str, int]:
    with open(path, "r") as f:
        token2id = json.load(f)

    assert len(token2id) > 0, "Length of vocabulary must be greater than zero"
    return token2id


def main():
    args = vars(cli())

    INPUT_FILE = args["input-file"]
    OUTPUT_FILE = args["output-file"]
    VOCAB_FILE = args["vocab"]
    COL_NAME = args["col_name"]

    token2id = read_vocab(VOCAB_FILE)
    df = read_input(INPUT_FILE, col_name=COL_NAME)

    tokenizer = LaTEXTokenizer(token2id=token2id, max_len=512)
    df["formula"] = df["formula"].apply(lambda x: [x])
    df["tokenized_formula"] = df["formula"].apply(
        tokenizer.tokenize, return_tensors=False, pad=False
    )

    save_output(OUTPUT_FILE, df)


# from project root
# python -m img2latex.data.preprocess ./data/interim/sample.test.csv \
# ./data/interim/tokens.test.csv --vocab ./data/processed/token2id.json
if __name__ == "__main__":
    main()
