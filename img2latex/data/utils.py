import pandas as pd
import json


def read_input(path: str, col_name: str = None) -> pd.DataFrame:
    # read input file or raise ValueError
    ext = path.strip().split(".")[-1]

    try:
        read_file = getattr(pd, f"read_{ext}")
        df = read_file(path)
        print(f"{df.shape=}")
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

    except Exception as e:
        raise e


def save_output(path: str, obj: object) -> None:
    if path.endswith(".json"):
        with open(path, "w") as f:
            json.dump(obj, f, indent=4)
    else:
        ext = path.strip().split(".")[-1]

        save = getattr(obj, f"to_{ext}")
        save(path, index=False)