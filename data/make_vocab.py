import argparse
import os
import json
import re
from collections import Counter
import pandas as pd


def cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Makes LaTEX vocabulary from a file. Outputs JSON file"
    )
    parser.add_argument(
        "input-file",
        type=str,
        help="Input file with formulas. Supports [csv, parquet, pickle]",
    )
    parser.add_argument(
        "output-folder",
        type=str,
        help="Output folder for token2id and id2token JSON files",
    )
    parser.add_argument(
        "--add-special",
        action=argparse.BooleanOptionalAction,
        help="Flag to add special tokens. SPECIAL_TOKENS = {'<UNK>': 0, '<SOS>': 1, '<PAD>': 2, '<EOS>': 3",
    )
    parser.add_argument(
        "--col-name",
        type=str,
        default=None,
        help="Column of DataFrame to be used. If set to None, DataFrame is required to have one column",
    )

    args = parser.parse_args()
    return args


def read_input(path: str, col_name: str = None) -> pd.DataFrame:
    # read input file or raise ValueError
    if path.endswith(".csv"):
        df = pd.read_csv(path)

    elif path.endswith(".parquet"):
        df = pd.read_parquet(path)

    elif path.endswith(".pkl"):
        df = pd.read_pickle(path)

    else:
        raise ValueError(
            f"Extension {path.split('.')[-1]} is not yet supported. Use one of [csv, parquet, pickle]"
        )

    # check if df has only one column
    if not col_name:
        assert (
            df.shape[1] == 1
        ), "DataFrame should have only one column or column to be used must be specified"
        col = df.columns.to_list()[0]
        df = df.rename({col: "formula"}, axis=1)
    # or name is specified
    else:
        df = df.rename({col_name: "formula"}, axis=1)

    return df


def save_output(
    token2id: dict[str, int], id2token: dict[int, str], path: str
) -> None:
    with open(f"{path}/token2id.json", "w") as f:
        json.dump(token2id, f, indent=4)

    with open(f"{path}/id2token.json", "w") as f:
        json.dump(id2token, f, indent=4)


def main():
    args = cli()

    INPUT_FILE = args.input_file
    OUTPUT_FOLDER = args.output_folder
    ADD_SPECIAL = args.add_special
    COL_NAME = args.col_name

    df = read_input(INPUT_FILE, COL_NAME)

    df["formula"] = df["formula"].apply(lambda x: x.replace(".", " . "))
    df["formula"] = df["formula"].apply(lambda x: re.sub("(\d)", r" \1", x))
    df["formula_tokenized"] = df["formula"].apply(lambda x: x.strip().split())

    words = df["formula_tokenized"].tolist()
    vocab = Counter([x for sublist in words for x in sublist])
    vocab = sorted(list(vocab.keys()))

    if ADD_SPECIAL:
        vocab = ["<UNK>", "<SOS>", "<PAD>", "<EOS>"] + vocab

    token2id = {k: i for i, k in enumerate(vocab)}
    id2token = {i: k for i, k in enumerate(vocab)}

    save_output(token2id, id2token, path=OUTPUT_FOLDER)


if __name__ == "__main__":
    main()
