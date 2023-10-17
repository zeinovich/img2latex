import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

from PIL import Image
from typing import Union
from .utils import read_input


class IM2LaTEX100K(Dataset):
    def __init__(
        self,
        file: str,
        transform: Union[T.Compose, None],
    ):
        super().__init__()
        self.file = file
        self.data = read_input(
            self.file, col_name="formula_tokenized"
        )  # "formula_tokenized" will be renamed into "formula"
        self.transform = transform

    def __getitem__(self, index) -> tuple[torch.Tensor]:
        row = self.data.iloc[index]
        tokens = row["formula"]
        img_path = row["image"]

        img = Image.open(img_path)

        if self.transform is not None:
            img = self.transform(img)

        tokens = torch.Tensor(tokens[0])

        return img, tokens

    def __len__(self) -> int:
        return self.data.shape[0]

    def __repr__(self) -> str:
        return f"IM2LaTEXDataset(file={self.file})"

    def __str__(self) -> str:
        return self.__repr__()
