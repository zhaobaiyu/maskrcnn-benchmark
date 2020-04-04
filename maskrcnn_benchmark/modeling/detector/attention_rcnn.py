# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity

from maskrcnn_benchmark.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads


class AttentionRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        self._device = torch.device(cfg.MODEL.DEVICE)
        super(AttentionRCNN, self).__init__()

        self.backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg, self.backbone.out_channels)
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)

        # conv layer after attention
        out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
        self.after_fusion_layer = nn.Conv2d(out_channels, out_channels, 1)
        nn.init.kaiming_uniform_(self.after_fusion_layer.weight, a=1)
        nn.init.constant_(self.after_fusion_layer.bias, 0)


    def _hash_sim(self, m1, m2, dim=0):
        # print('hash', end='')
        # m1: c * h * w
        assert m1.size() == m2.size()
        c, h, w = m1.size()[0], m1.size()[1], m1.size()[2]

        # new_m1: c * (h+2) * (w+2)
        new_m1 = F.pad(m1.unsqueeze(0), (1, 1, 1, 1), mode='replicate').squeeze(0)
        new_m2 = F.pad(m2.unsqueeze(0), (1, 1, 1, 1), mode='replicate').squeeze(0)
        conv_filter = torch.ones((1, c, 3, 3)).to(self._device)
        # avg_m1: 1 * h * w
        avg_m1 = F.conv2d(new_m1.unsqueeze(0), conv_filter, padding=0).squeeze(0) / conv_filter.numel()
        avg_m2 = F.conv2d(new_m2.unsqueeze(0), conv_filter, padding=0).squeeze(0) / conv_filter.numel()

        '''
        # method 1:
        # res: h * w
        res = torch.zeros(m1.size()[1], m1.size()[2], device='cuda')
        for i in range(m1.size()[1]):
            for j in range(m1.size()[2]):
                # tmp1: 1024 * 3 * 3
                tmp1 = new_m1[:, i:i+3, j:j+3] >= avg_m1[0, i, j]
                tmp2 = new_m2[:, i:i+3, j:j+3] >= avg_m2[0, i, j]
                res[i, j] = (tmp1 == tmp2).sum()
                
        # method 2:        
        # tmp1: h * w * c * 3 * 3
        tmp1 = torch.zeros([m1.size()[1], m1.size()[2], m1.size()[0], 3, 3])
        tmp2 = tmp1.clone()
        for i in range(m1.size()[1]):
            for j in range(m1.size()[2]):
                tmp1[i, j, :, :, :] = new_m1[:, i:i+3, j:j+3]
                tmp2[i, j, :, :, :] = new_m2[:, i:i+3, j:j+3]
        tmp1 = tmp1 >= avg_m1.view([avg_m1.size()[1], avg_m1.size()[2], 1, 1, 1])
        tmp2 = tmp2 >= avg_m2.view([avg_m2.size()[1], avg_m2.size()[2], 1, 1, 1])
        res = (tmp1 == tmp2).sum(dim=[2, 3, 4])
        '''

        # correct method
        tmp1 = torch.nn.functional.unfold(new_m1.unsqueeze(0), kernel_size=[3, 3]).squeeze(0).view(-1, h, w)
        tmp2 = torch.nn.functional.unfold(new_m2.unsqueeze(0), kernel_size=[3, 3]).squeeze(0).view(-1, h, w)
        res = ((tmp1 >= avg_m1) == (tmp2 >= avg_m2)).sum(dim=0, dtype=torch.float)
        return res

    def new_features(self, features, batches_nearby_features, sim):
        new_features_list = []
        for i, nearby_features in enumerate(batches_nearby_features):
            weights_list = []
            for ii, nearby_feature in enumerate(nearby_features):
                # print(features[i].size())
                weights_list.append(sim(features[i], nearby_feature, dim=0).unsqueeze(dim=0))
            weights = torch.nn.functional.softmax(torch.cat(weights_list, dim=0), dim=0).unsqueeze(dim=1)
            # new_features_list.append(torch.sum(weights * nearby_features, dim=0, keepdim=True))
            new_features = self.after_fusion_layer(torch.sum(weights * nearby_features, dim=0, keepdim=True) + features[i])
            new_features_list.append(new_features)
        return torch.cat(new_features_list, dim=0)


    def forward(self, images, batches_nearby_images, targets=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images)
        features = self.backbone(images.tensors)

        if batches_nearby_images is not None:
            fpn_num = len(features)
            batches_nearby_features = [[] for _ in range(fpn_num)]
            # batches_nearby_images is a list, each is one in a batch
            # each is a ImageList with [4, 3, 768, 1344], 4 is num of nearby frames
            #print('-----------debug attention-rcnn, batches-----------')
            #print(len(batches_nearby_images), batches_nearby_images[0].tensors.size())
            #print(len(batches_nearby_features), batches_nearby_features[0].size())
            
            for nearby_images in batches_nearby_images:
                nearby_images = to_image_list(nearby_images)
                nearby_features = self.backbone(nearby_images.tensors)
                for i in range(fpn_num):
                    batches_nearby_features[i].append(nearby_features[i])
            # batches_nearby_features is a list, each is one is a FPN level
            # [[(4,256,100,200),(other batch)],[(4,256,50,100),()],[(4,256,25,50),()],[(4,256,12,25),()],...]
            # 4 is num of nearby frames

            post_features = []
            for i in range(fpn_num):
                post_features.append(self.new_features(features[i], batches_nearby_features[i], sim=self._hash_sim))
            features = tuple(post_features)


        proposals, proposal_losses = self.rpn(images, features, targets)
        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(features, proposals, targets)
        else:
            # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}

        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses

        return result
