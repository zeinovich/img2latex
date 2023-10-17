import argparse
import re
from collections import Counter

from .utils import read_input, save_output


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


def main():
    args = vars(cli())
    print(args)
    INPUT_FILE = args["input-file"]
    OUTPUT_FOLDER = args["output-folder"]
    ADD_SPECIAL = args["add_special"]
    COL_NAME = args["col_name"]

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

    save_output(path=f"{OUTPUT_FOLDER}/token2id.json", obj=token2id)
    save_output(path=f"{OUTPUT_FOLDER}/id2token.json", obj=id2token)


if __name__ == "__main__":
    main()
