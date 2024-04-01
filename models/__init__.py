# interface for each model module


import abc
import torch
from PIL import Image


class Preprocessor(abc.ABC):
    @staticmethod
    @abc.abstractmethod
    def preprocess(image: Image) -> torch.Tensor:
        raise NotImplementedError


class Model(abc.ABC):
    def __init__(self, device: str, config: dict) -> None:
        self.device: str = device
        self.config: dict = config
        self.model: torch.nn.Module = None
        self.optimizer: torch.optim.Optimizer = None
        self.loss_fn: torch.nn.Module = None
    
    def __str__(self):
        return self.__class__.__name__.replace('Model', '')

    def set_device(self, device: str):
        self.device = device
        
    @abc.abstractmethod
    def to_device(self, device: str | None = None):
        raise NotImplementedError

    @abc.abstractmethod
    def load_model_and_optimizer(self, checkpoint):
        raise NotImplementedError

    @abc.abstractmethod
    def save_model_and_optimizer(self, name: str):
        raise NotImplementedError

    @abc.abstractmethod
    def get_embedding(self, image):
        raise NotImplementedError
    
    @abc.abstractmethod
    def get_classification(self, image, label):
        raise NotImplementedError

    @abc.abstractmethod
    def get_loss(self, image, label):
        raise NotImplementedError
