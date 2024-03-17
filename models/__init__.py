# interface for each model module


import abc


class Model(abc.ABC):
    def __init__(self) -> None:
        self.model = None
        self.optimizer = None
        self.train_dl = None
        self.test_dl = None
        self.val_dl = None
        self.loss_fn = None

    def set_device(self, device: str):
        self.device = device

    @abc.abstractmethod
    def load_model_and_optimizer(self, checkpoint):
        raise NotImplementedError

    @abc.abstractmethod
    def save_model_and_optimizer(self, name: str):
        raise NotImplementedError
            
    @abc.abstractmethod
    def setup_dataloaders(self):
        raise NotImplementedError

    @abc.abstractmethod
    def preprocess(self, image):
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
