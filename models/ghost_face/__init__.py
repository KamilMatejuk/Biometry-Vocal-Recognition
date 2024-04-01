import torch
from PIL import Image
from torchvision import transforms

from models.ghost_face.src.ghostnetv2_pytorch.model.ghostnetv2_torch import ghostnetv2

from models import Model, Preprocessor
from loggers import ghost_face_logger as logger



class GhostFacePreprocessorTrain(Preprocessor):
    @staticmethod
    def preprocess(image: Image) -> torch.Tensor:
        # image = image.convert('L')
        trans = transforms.Compose([
            transforms.RandomResizedCrop(size=(128, 128), scale=(0.08, 1.0), ratio=(0.75, 1.3333), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=(0.6, 1.4), contrast=(0.6, 1.4), saturation=(0.6, 1.4)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250]),
        ])
        return trans(image)


class GhostFacePreprocessorTest(Preprocessor):
    @staticmethod
    def preprocess(image: Image) -> torch.Tensor:
        image = image.convert('L')
        trans = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250]),
        ])
        return trans(image)


class GhostFaceModel(Model):
    def __init__(self, device: str) -> None:
        super().__init__(device)
        self.num_classes = 10177
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.model = ghostnetv2(num_classes=self.num_classes, width=1.0, dropout=0.0, args=None)
        self.optimizer = torch.optim.SGD(params=self.model.parameters(),
            lr=1e-2, momentum=0.9, weight_decay=1e-4)
        self.to_device()

    def to_device(self, device: str | None = None):
        if device is None: device = self.device
        self.model.to(device)
        self.loss_fn.to(device)

    def load_model_and_optimizer(self, checkpoint_file: str):
        logger.info(f'Loading from file {checkpoint_file}')
        checkpoint = torch.load(checkpoint_file, map_location=torch.device(self.device))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        pass

    def save_model_and_optimizer(self, name: str):
        torch.save(obj={
            'model_state_dict': self.model.cpu().state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, f=name)
        self.to_device()

    def get_embedding(self, image):
        # forward without self.classifier call
        x = self.conv_stem(image)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.blocks(x)
        x = self.global_pool(x)
        x = self.conv_head(x)
        x = self.act2(x)
        x = x.view(x.size(0), -1)
        return x
    
    def get_classification(self, image, label):
        return self.model(image)

    def get_loss(self, image, label):
        return self.loss_fn(image, label)
