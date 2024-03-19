import torch
from PIL import Image
from torchvision import transforms

from models.arc_face.src.models import FocalLoss, ResNetFace, IRBlock, ArcMarginProduct

from models import Model, Preprocessor
from loggers import arc_face_logger as logger


class ArcFacePreprocessorTrain(Preprocessor):
    @staticmethod
    def preprocess(image: Image) -> torch.Tensor:
        image = image.convert('L')
        trans = transforms.Compose([
            transforms.Resize((128, 128)),
            # transforms.RandomCrop((128, 128)), https://github.com/ronghuaiyang/arcface-pytorch/blob/master/data/dataset.py#L31
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        return trans(image)


class ArcFacePreprocessorTest(Preprocessor):
    @staticmethod
    def preprocess(image: Image) -> torch.Tensor:
        image = image.convert('L')
        trans = transforms.Compose([
            transforms.Resize((128, 128)),
            # transforms.CenterCrop((128, 128)), https://github.com/ronghuaiyang/arcface-pytorch/blob/master/data/dataset.py#L31
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        return trans(image)


class ArcFaceModel(Model):
    def __init__(self, device: str) -> None:
        super().__init__(device)
        self.num_classes = 10177
        self.loss_fn = FocalLoss(2)
        self.model = ResNetFace(IRBlock, [2, 2, 2, 2], use_se=False).to(self.device)
        self.metric_fc = ArcMarginProduct(512, self.num_classes, s=30, m=0.5, easy_margin=False).to(self.device)
        self.optimizer = torch.optim.SGD(
            [{'params': self.model.parameters()}, {'params': self.metric_fc.parameters()}],
            lr=1e-1, weight_decay=5e-4)

    def load_model_and_optimizer(self, checkpoint):
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.metric_fc.load_state_dict(checkpoint['metric_fc_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def save_model_and_optimizer(self, name: str):
        torch.save(obj={
            'model_state_dict': self.model.cpu().state_dict(),
            'metric_fc_state_dict': self.metric_fc.cpu().state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, f=name)

    def get_embedding(self, image):
        return self.model(image)
    
    def get_classification(self, image, label):
        feature = self.model(image)
        return self.metric_fc(feature, label)

    def get_loss(self, image, label):
        return self.loss_fn(image, label)
