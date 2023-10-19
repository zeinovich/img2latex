import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

from PIL import Image
from typing import Union
from .utils import read_input, prepare_csv_array


class IM2LaTEX100K(Dataset):
    def __init__(
        self,
        file: str,
        image_folder: str,
        vocab_len: int,
        transform: Union[T.Compose, None],
    ):
        super().__init__()
        self._file = file
        self._data = read_input(
            self._file, col_name="formula_tokenized"
        )  # "formula_tokenized" will be renamed into "formula"
        if file.endswith(".csv"):
            self._data["formula"] = self._data["formula"].apply(
                lambda x: prepare_csv_array(x)
            )
        self._transform = transform
        self._vocab_len = vocab_len
        self._image_folder = image_folder

    def __getitem__(self, index) -> tuple[torch.Tensor]:
        row = self._data.iloc[index]
        tokens = row["formula"]
        img_path = f"{self._image_folder}/{row['image']}"

        img = Image.open(img_path)

        if self._transform is not None:
            img = self._transform(img)

        tokens = torch.Tensor(tokens).long()

        return img, tokens

    def __len__(self) -> int:
        return self._data.shape[0]

    def __repr__(self) -> str:
        return f"IM2LaTEXDataset(file={self._file}, image_folder={self._image_folder})"

    def __str__(self) -> str:
        return self.__repr__()
