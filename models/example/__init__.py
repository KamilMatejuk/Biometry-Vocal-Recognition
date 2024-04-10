import torch
from PIL import Image
from torchvision import transforms

from models.example.src.ghostnetv2_pytorch.model.ghostnetv2_torch import ghostnetv2

from models import Model, Preprocessor
from loggers import example_logger as logger



class ExamplePreprocessorTrain(Preprocessor):
    @staticmethod
    def preprocess(image: Image) -> torch.Tensor:
        '''Preprocess one instance for training'''
        # trans = transforms.Compose([
        #     transforms.RandomResizedCrop(size=(128, 128), scale=(0.08, 1.0), ratio=(0.75, 1.3333), interpolation=transforms.InterpolationMode.BICUBIC),
        #     transforms.RandomHorizontalFlip(p=0.5),
        #     transforms.ColorJitter(brightness=(0.6, 1.4), contrast=(0.6, 1.4), saturation=(0.6, 1.4)),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250]),
        # ])
        # return trans(image)
        raise NotImplementedError


class ExamplePreprocessorTest(Preprocessor):
    @staticmethod
    def preprocess(image: Image) -> torch.Tensor:
        '''Preprocess one instance for testing'''
        # trans = transforms.Compose([
        #     transforms.Resize((128, 128)),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250]),
        # ])
        # return trans(image)
        raise NotImplementedError


class ExampleModel(Model):
    def __init__(self, device: str, config: dict) -> None:
        super().__init__(device, config)
        '''Create instance of model, optimizer and loss'''
        # self.num_classes = self.config['num_classes']
        # self.loss_fn = torch.nn.CrossEntropyLoss()
        # self.model = ghostnetv2(num_classes=self.num_classes, width=self.config['width'], dropout=self.config['dropout'], args=None)
        # self.optimizer = torch.optim.SGD(params=self.model.parameters(),
        #     lr=float(self.config['lr']), momentum=self.config['momentum'], weight_decay=float(self.config['wd']))
        self.to_device()

    def to_device(self, device: str | None = None):
        '''Move all possible components to device (e.g. model, loss)'''
        # if device is None: device = self.device
        # self.model.to(device)
        # self.loss_fn.to(device)
        raise NotImplementedError

    def load_model_and_optimizer(self, checkpoint_file: str):
        '''Load state dicts for all required components'''
        # logger.info(f'Loading from file {checkpoint_file}')
        # checkpoint = torch.load(checkpoint_file, map_location=torch.device(self.device))
        # self.model.load_state_dict(checkpoint['model_state_dict'])
        # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        raise NotImplementedError

    def save_model_and_optimizer(self, name: str):
        '''Save state dicts for all required components'''
        # torch.save(obj={
        #     'model_state_dict': self.model.cpu().state_dict(),
        #     'optimizer_state_dict': self.optimizer.state_dict(),
        # }, f=name)
        # self.to_device()
        raise NotImplementedError

    def get_embedding(self, image):
        '''Get embedding from model'''
        # with torch.no_grad():
        #     # forward without self.classifier call
        #     x = self.model.conv_stem(image)
        #     x = self.model.bn1(x)
        #     x = self.model.act1(x)
        #     x = self.model.blocks(x)
        #     x = self.model.global_pool(x)
        #     x = self.model.conv_head(x)
        #     x = self.model.act2(x)
        #     x = x.view(x.size(0), -1)
        #     return x.flatten().cpu()
        raise NotImplementedError
            
    
    def get_classification(self, image, label):
        '''Get classification from model'''
        # return self.model(image)
        raise NotImplementedError

    def get_loss(self, image, label):
        '''Get loss'''
        # return self.loss_fn(image, label)
        raise NotImplementedError
