import torch

from .src.models.focal_loss import FocalLoss
from .src.models.resnet import ResNetFace, IRBlock
from .src.models.metrics import ArcMarginProduct
from .. import Model


class ArcFaceModel(Model):
    def __init__(self) -> None:
        super().__init__()
        self.metric_fc = None
        self.loss_fn = FocalLoss(2)

    def load_model_and_optimizer(self, checkpoint, num_classes):
        # create model (and other if neccesary) instance from checkpoint
        self.model = ResNetFace(IRBlock, [2, 2, 2, 2], use_se=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.metric_fc = ArcMarginProduct(512, num_classes, s=30, m=0.5, easy_margin=False)
        self.metric_fc.load_state_dict(checkpoint['metric_fc_state_dict'])
        self.optimizer = torch.optim.SGD(
            [{'params': self.model.parameters()}, {'params': self.metric_fc.parameters()}],
            lr=1e-1, weight_decay=5e-4)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def save_model_and_optimizer(self, name: str):
        torch.save(obj={
            'model_state_dict': self.model.state_dict(),
            'metric_fc_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, f=name)
            
    def setup_dataloaders(self):
        pass

    def preprocess(self, image):
        pass

    def get_embedding(self, image):
        return self.model(image)
    
    def get_classification(self, image, label):
        feature = self.model(image)
        return self.metric_fc(feature, label)

    def get_loss(self, image, label):
        return self.loss_fn(image, label)
