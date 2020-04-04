import os

import torch
import torch.utils.data
from PIL import Image
import sys

import xml.etree.ElementTree as ET

from maskrcnn_benchmark.structures.bounding_box import BoxList


class DETDataset(torch.utils.data.Dataset):

    CLASSES = ('__background__', 'airplane', 'antelope', 'bear', 'bicycle', 'bird', 'bus', 'car', 'cattle', 'dog',
               'domestic cat', 'elephant', 'fox', 'giant panda', 'hamster', 'horse', 'lion', 'lizard', 'monkey',
               'motorcycle', 'rabbit', 'red panda', 'sheep', 'snake', 'squirrel', 'tiger', 'train', 'turtle',
               'watercraft', 'whale', 'zebra')

    CID_TO_CLASSES = {'n02691156':'airplane', 'n02419796':'antelope', 'n02131653':'bear', 'n02834778':'bicycle',
                      'n01503061':'bird', 'n02924116':'bus', 'n02958343':'car', 'n02402425':'cattle', 'n02084071':'dog',
                      'n02121808':'domestic cat', 'n02503517':'elephant', 'n02118333':'fox', 'n02510455':'giant panda',
                      'n02342885':'hamster', 'n02374451':'horse', 'n02129165':'lion', 'n01674464':'lizard',
                      'n02484322':'monkey', 'n03790512':'motorcycle', 'n02324045':'rabbit', 'n02509815':'red panda',
                      'n02411705':'sheep', 'n01726692':'snake', 'n02355227':'squirrel', 'n02129604':'tiger',
                      'n04468005':'train', 'n01662784':'turtle', 'n04530566':'watercraft', 'n02062744':'whale',
                      'n02391049':'zebra'}

    def __init__(self, data_dir, split, imgsetname, frame_selection='each1', transforms=None):
        """

        :param data_dir:
        :param split: split: 'train', 'val', or 'test'
        :param transforms:
        :param frame_selection:
        """

        # root dir: /path/to/ILSVRC
        self.root = data_dir
        # split: 'train', 'val' or 'test'
        # self.split = split
        self.transforms = transforms

        self._data_dir = os.path.join(self.root, 'Data', 'DET')
        self._anno_dir = os.path.join(self.root, 'Annotations', 'DET')
        self._imgpath = os.path.join(self._data_dir, '{}.JPEG')
        self._annopath = os.path.join(self._anno_dir, '{}.xml')

        self._imgsetpath = os.path.join(self.root, "ImageSets", imgsetname)
        self.ids = []
        with open(self._imgsetpath) as f:
            tmp_ids = f.readlines()
        for x in tmp_ids:
            path = x.strip("\n")[:-2]
            if self._objs_exist(path):
                self.ids.append(path)

        self.id_to_img_map = {k:v for k, v in enumerate(self.ids)}

        cls = DETDataset.CLASSES
        self.class_to_ind = dict(zip(cls, range(len(cls))))
        self.categories = dict(zip(range(len(cls)), cls))

    def _objs_exist(self, path):
        imgpath = self._imgpath.format(path)
        annopath = self._annopath.format(path)
        if not os.path.exists(imgpath) or not os.path.exists(annopath):
            return False
        return True if ET.parse(annopath).getroot().find('object') else False

    def __getitem__(self, index):
        img_id = self.ids[index]
        img = Image.open(self._imgpath.format(img_id)).convert("RGB")

        target = self.get_groundtruth(index)
        target = target.clip_to_image(remove_empty=True)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, None, target, index

    def __len__(self):
        return len(self.ids)

    def get_groundtruth(self, index):
        img_id = self.ids[index]
        anno = ET.parse(self._annopath.format(img_id)).getroot()
        anno = self._preprocess_annotation(anno)

        height, width = anno['im_info']
        target = BoxList(anno['boxes'], (width, height), mode='xyxy')
        target.add_field('labels', anno['labels'])
        target.add_field("difficult", anno["difficult"])
        return target

    def _preprocess_annotation(self, target):
        boxes = []
        gt_classes = []
        difficult_boxes = []
        TO_REMOVE = 1

        for obj in target.iter('object'):
            if obj.find('name').text.lower().strip() not in DETDataset.CID_TO_CLASSES:
                continue
            difficult = False
            name = DETDataset.CID_TO_CLASSES[obj.find('name').text.lower().strip()]
            bb = obj.find('bndbox')
            box = [
                bb.find('xmin').text,
                bb.find('ymin').text,
                bb.find('xmax').text,
                bb.find('ymax').text,
            ]
            bndbox = tuple(
                map(lambda x: x - TO_REMOVE, list(map(int, box)))
            )

            boxes.append(bndbox)
            gt_classes.append(self.class_to_ind[name])
            difficult_boxes.append(difficult)

        size = target.find('size')
        im_info = tuple(map(int, (size.find('height').text, size.find('width').text)))

        res = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(gt_classes),
            'difficult': torch.tensor(difficult_boxes),
            'im_info': im_info,
        }

        return res

    def _has_anno(self, img_id):
        anno = ET.parse(self._annopath.format(img_id)).getroot()
        if anno.find('object'):
            return True
        else:
            return False

    def get_img_info(self, index):
        img_id = self.ids[index]
        anno = ET.parse(self._annopath.format(img_id)).getroot()
        size = anno.find('size')
        im_info = tuple(map(int, (size.find('height').text, size.find('width').text)))
        return {'height': im_info[0], 'width': im_info[1]}

    def map_class_id_to_class_name(self, class_id):
        return DETDataset.CLASSES[class_id]
