import torch
import torch.nn as nn
from torchvision.models.detection import retinanet_resnet50_fpn
from dataloader import KfashionDataset
from torch.utils.data import DataLoader
from torchvision.transforms import transforms as T
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def collate_fn(batch):
    return tuple(zip(*batch))

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        # (역자주: 학습시 50% 확률로 학습 영상을 좌우 반전 변환합니다)
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


img_dir = r'C:\Users\jsk\Documents\data\train'
anno_dir = r'C:\Users\jsk\Documents\data'
set_name = 'train'
dataset = KfashionDataset(img_dir, anno_dir, set_name, transforms=get_transform(True))
num_classes = len(dataset.category)
print(f'num_classes: {num_classes}')

dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

model = retinanet_resnet50_fpn(True)
num_anchor = model.anchor_generator.num_anchors_per_location()[0]
in_channels = model.head.classification_head.cls_logits.in_channels
model.head.classification_head = RetinaNetClassificationHead(in_channels, num_anchor, num_classes)

optim = torch.optim.SGD(model.parameters(), lr=1e-4)

model.train()
for img, targets in dataloader:
    optim.zero_grad()
    out = model(img, targets)

    cls_loss = out['classification']
    reg_loss = out['bbox_regression']

    loss = cls_loss + reg_loss
    print(f'loss: {loss}')

    loss.backward()
    optim.step()