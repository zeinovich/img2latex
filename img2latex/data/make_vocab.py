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


def make_vocab_func(
    input_file: str, col_name: str, add_special: bool, output_folder: str
) -> None:
    df = read_input(input_file, col_name)

    df["formula"] = df["formula"].apply(lambda x: x.replace(".", " . "))
    df["formula"] = df["formula"].apply(lambda x: re.sub("(\d)", r" \1", x))
    df["formula_tokenized"] = df["formula"].apply(lambda x: x.strip().split())

    words = df["formula_tokenized"].tolist()
    vocab = Counter([x for sublist in words for x in sublist])
    vocab = sorted(list(vocab.keys()))

    if add_special:
        vocab = ["<UNK>", "<SOS>", "<PAD>", "<EOS>"] + vocab

    token2id = {k: i for i, k in enumerate(vocab)}
    id2token = {i: k for i, k in enumerate(vocab)}

    save_output(path=f"{output_folder}/token2id.json", obj=token2id)
    save_output(path=f"{output_folder}/id2token.json", obj=id2token)


def main():
    args = vars(cli())
    args = {k.replace("-", "_"): v for k, v in args.items()}
    print(args)
    make_vocab_func(**args)


if __name__ == "__main__":
    main()
