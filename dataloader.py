from torch.utils.data import Dataset
from pycocotools.coco import COCO
import os
import cv2
import torch
from PIL import Image


class KfashionDataset(Dataset):
    def __init__(self, img_dir, anno_dir, set_name='train', transforms=None):
        self.img_dir = img_dir
        self.anno_dir = anno_dir
        self.transforms = transforms

        self.coco = COCO(os.path.join(self.anno_dir, f'{set_name}.json'))

        self.image_ids = self.coco.getImgIds()
        self.category = self.coco.loadCats(self.coco.getCatIds())

    def __getitem__(self, ids):

        img = self.get_image(ids)
        target = self.get_annotaion(ids)

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.image_ids)

    def get_image(self, ids):
        img_info = self.coco.loadImgs(self.image_ids[ids])[0]
        key = list(img_info['file_name'])[0]
        img_path = os.path.join(self.img_dir, key, img_info['file_name'])
        img = Image.open(img_path)
        # img = cv2.imread(img_path)

        return img

    def get_annotaion(self, ids):
        anno_ids = self.coco.getAnnIds(imgIds=self.image_ids[ids], iscrowd=False)
        target = {}
        boxes = []
        labels = []

        # some images appear to miss annotations
        if len(anno_ids) == 0:
            target['boxes'] = torch.as_tensor([[10, 10, 20, 20]], dtype=torch.float32)
            target['labels'] = torch.as_tensor([[0]], dtype=torch.float32)
            return target

        coco_annos = self.coco.loadAnns(anno_ids)

        for coco_anno in coco_annos:
            bbox = self.bbox_transform(coco_anno['bbox'])
            boxes.append(bbox)
            labels.append(coco_anno['category_id'])

        target['boxes'] = torch.as_tensor(boxes, dtype=torch.float32)
        target['labels'] = torch.as_tensor(labels, dtype=torch.int64)

        return target

    def bbox_transform(self, bbox):
        bbox[2] = bbox[0] + bbox[2]
        bbox[3] = bbox[1] + bbox[3]
        return bbox


if __name__ == '__main__':

    img_dir = r'C:\Users\jsk\Documents\data\train'
    anno_dir = r'C:\Users\jsk\Documents\data'
    set_name = 'train'
    dataset = KfashionDataset(img_dir, anno_dir, set_name)

    from torch.utils.data import DataLoader

    dataloader = DataLoader(dataset, batch_size=1)

    for sample in dataloader:
        img = sample['img']
        anno = sample['anno']
        break;