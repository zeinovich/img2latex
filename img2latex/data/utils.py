import pandas as pd
import json


def read_input(path: str, col_name: str = None) -> pd.DataFrame:
    # read input file or raise ValueError
    ext = path.strip().split(".")[-1]

    if ext == "json":
        with open(path, "r") as f:
            df = json.load(f)

    else:
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

        except Exception as e:
            raise e

    return df


def save_output(path: str, obj: object) -> None:
    if path.endswith(".json"):
        with open(path, "w") as f:
            json.dump(obj, f, indent=4)
    else:
        ext = path.strip().split(".")[-1]

        # get pandas to_{extension}, ie to_csv, method
        save = getattr(obj, f"to_{ext}")
        save(path, index=False)


def prepare_csv_array(row: str) -> list[int]:
    row = row.replace("[", "")
    row = row.replace("]", "")
    row = row.strip().split(",")
    row = list(map(int, row))
    return row
