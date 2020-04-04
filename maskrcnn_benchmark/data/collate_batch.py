# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from maskrcnn_benchmark.structures.image_list import to_image_list


class BatchCollator(object):
    """
    From a list of samples from the dataset,
    returns the batched images and targets.
    This should be passed to the DataLoader
    """

    def __init__(self, size_divisible=0):
        self.size_divisible = size_divisible

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        images = to_image_list(transposed_batch[0], self.size_divisible)

        if not None in transposed_batch[1]:
        # if transposed_batch[1][0] is not None:
            # transposed_batch[1] is tuple, len(transposed_batch[1]) = batch_size
            batches_nearby_imgs = []
            for nearby_imgs in transposed_batch[1]:
                batches_nearby_imgs.append(to_image_list(nearby_imgs, self.size_divisible))
        else:
            batches_nearby_imgs = None
            
        targets = transposed_batch[2]
        img_ids = transposed_batch[3]
        return images, batches_nearby_imgs, targets, img_ids

class BBoxAugCollator(object):
    """
    From a list of samples from the dataset,
    returns the images and targets.
    Images should be converted to batched images in `im_detect_bbox_aug`
    """

    def __call__(self, batch):
        return list(zip(*batch))

