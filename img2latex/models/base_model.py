# Abstract Base Class for model

import torch
import torch.nn as nn

from abc import ABC, abstractmethod, abstractproperty


class BaseIm2SeqModel(ABC, nn.Module):
    @abstractmethod
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractproperty
    def device(self) -> torch.device:
        pass

    @abstractproperty
    def img_size(self) -> int:
        pass

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)
