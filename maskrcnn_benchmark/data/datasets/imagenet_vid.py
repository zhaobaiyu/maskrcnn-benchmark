import os
import random
from collections import defaultdict

import torch
import torch.utils.data
from PIL import Image
import sys

import xml.etree.ElementTree as ET

from maskrcnn_benchmark.structures.bounding_box import BoxList


class VIDDataset(torch.utils.data.Dataset):

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

    def __init__(self, data_dir, split, imgsetname, frame_selection='each2', transforms=None):
        """

        :param data_dir:
        :param split: split: 'train', 'val', or 'test'
        :param transforms:
        :param frame_selection:
        """

        # root dir: /path/to/ImageVID
        self.root = data_dir
        # split: 'train', 'val' or 'test'
        # self.split = split
        self.transforms = transforms

        self._data_dir = os.path.join(self.root, 'Data', 'VID')
        self._anno_dir = os.path.join(self.root, 'Annotations', 'VID')
        self._imgpath = os.path.join(self._data_dir, '{}.JPEG')
        self._annopath = os.path.join(self._anno_dir, '{}.xml')

        # config about frame selection
        self.frame_selection = frame_selection
        if self.frame_selection == 'pre5':
            self.frame_slice = slice(5, None)
            self.get_needed_range = lambda x: range(x-5, x)
            self.nearby_num = 5
        elif self.frame_selection == 'sub5':
            self.frame_slice = slice(None, -5)
            self.get_needed_range = lambda x: range(x+1, x+6)
            self.nearby_num = 5
        elif self.frame_selection == 'each2':
            self.frame_slice = slice(2, -2)
            self.get_needed_range = lambda x: [i for i in (x-2, x-1, x+1, x+2)]
            self.nearby_num = 4
        elif self.frame_selection == 'each1':
            self.frame_slice = slice(1, -1)
            self.get_needed_range = lambda x: [i for i in (x-1, x+1)]
            self.nearby_num = 2

        self._imgsetpath = os.path.join(self.root, "ImageSets", imgsetname)
        self.ids = []
        with open(self._imgsetpath) as f:
            tmp_ids = f.readlines()
        if split == 'train':
            for x in tmp_ids:
                path = os.path.join(x.split()[0], '{:0>6}'.format(x.split()[2]))
                if self._objs_exist(path):
                    self.ids.append(path)
        elif split == 'val':
            tiny_val = 0
            if tiny_val:
                split_dict = defaultdict(list)
                splits = set()
                for path in tmp_ids:
                    split_path = os.path.split(path.split()[0])[0]
                    split_dict[split_path].append(path)
                    splits.add(split_path)
                splits = random.sample(list(splits), tiny_val)
                tmp_ids = []
                for split_path in splits:
                    for path in split_dict[split_path]:
                        tmp_ids.append(path)
            for x in tmp_ids:
                path = x.split()[0]
                if self._objs_exist(path):
                    self.ids.append(path)

        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}

        cls = VIDDataset.CLASSES
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
        img_path = self._imgpath.format(img_id)
        img = Image.open(img_path).convert('RGB')

        img_id_prefix = img_id[:-6]
        img_id_num = int(img_id[-6:])

        imgs_needed_paths = [self._imgpath.format('{}{:0>6d}'.format(img_id_prefix, x)) for x in self.get_needed_range(img_id_num)]
        nearby_imgs = [Image.open(x).convert('RGB') for x in imgs_needed_paths if os.path.exists(x)]

        target = self.get_groundtruth(index)
        target = target.clip_to_image(remove_empty=True)

        if self.transforms is not None:
            img, target = self.transforms(img, target)
            tmp = nearby_imgs
            nearby_imgs = []
            for img_needed in tmp:
                tmp_img, _ = self.transforms(img_needed, target)
                nearby_imgs.append(tmp_img)
        if len(nearby_imgs) < self.nearby_num:
            nearby_imgs = None
        return img, nearby_imgs, target, index

    def __len__(self):
        return len(self.ids)

    def get_groundtruth(self, index):
        img_id = self.ids[index]
        anno = ET.parse(self._annopath.format(img_id)).getroot()
        anno = self._process_annotation(anno)

        height, width = anno['im_info']
        target = BoxList(anno['boxes'], (width, height), mode='xyxy')
        target.add_field('labels', anno['labels'])
        target.add_field("difficult", anno["difficult"])
        return target

    def _process_annotation(self, target):
        boxes = []
        gt_classes = []
        difficult_boxes = []
        TO_REMOVE = 1

        for obj in target.iter('object'):
            difficult = False
            name = VIDDataset.CID_TO_CLASSES[obj.find('name').text.lower().strip()]
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

    def _get_ids(self, image_set):
        ids = []
        if image_set == 'train':
            snippets_sets = os.listdir(self._data_dir)
            for snippets_set in snippets_sets:
                snippets_set_path = os.path.join(self._data_dir, snippets_set)
                snippets = os.listdir(snippets_set_path)
                # for this model, we use the previous frames to predict the feature of current frame
                for snippet in snippets:
                    snippet_path = os.path.join(snippets_set_path, snippet)

                    for frame in sorted(os.listdir(snippet_path))[self.frame_slice]:
                        tmp_img_id = os.path.join(snippets_set, snippet, frame[:-5])
                        if self._has_anno(tmp_img_id):
                            ids.append(tmp_img_id)
                    # ids.extend([os.path.join(snippets_set, snippet, frame)[:-5] for frame in sorted(os.listdir(snippet_path))[self.frame_slice]])

        else:
            snippets = os.listdir(self._data_dir)
            for snippet in snippets:
                snippet_path = os.path.join(self._data_dir, snippet)

                for frame in sorted(os.listdir(snippet_path))[self.frame_slice]:
                    tmp_img_id = os.path.join(snippet, frame[:-5])
                    if self._has_anno(tmp_img_id):
                        ids.append(tmp_img_id)
                # ids.extend([os.path.join(snippet, frame)[:-5] for frame in sorted(os.listdir(snippet_path))[self.frame_slice]])
        return ids

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
        return VIDDataset.CLASSES[class_id]
