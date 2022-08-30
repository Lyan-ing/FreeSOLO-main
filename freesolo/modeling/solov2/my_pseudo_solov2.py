# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/FreeSOLO/blob/main/LICENSE

# -------------------------------------------------------------------------
# Copyright (c) 2019 the AdelaiDet authors
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Modified by Xinlong Wang
# -------------------------------------------------------------------------

from skimage import color
import torch
import torch.nn.functional as F

from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.structures import ImageList

from .my_solov2 import My_SOLOv2
from .utils import point_nms, matrix_nms, get_images_color_similarity
from .loss import dice_loss, FocalLoss
from torchvision.utils import draw_bounding_boxes
import torchshow as ts


@META_ARCH_REGISTRY.register()
class My_PseudoSOLOv2(My_SOLOv2):
    def forward(
            self, batched_inputs, branch="supervised", given_proposals=None, val_mode=False
    ):

        # original_images = [x["image"].to(self.device) for x in batched_inputs]  # 经过数据增强的图
        # tmp_img = batched_inputs[0]["image"]
        # tmp_box = batched_inputs[0]['instances'].gt_boxes[0]
        # result = draw_bounding_boxes(tmp_img.cpu(), tmp_box.tensor, colors=["red"], width=2)
        # ts.save(result, './test1.png')
        images, depth_prediction, original_images = self.preprocess_image(batched_inputs)  # Normalize, pad and batch the input images.变为统一大小

        features = self.backbone(images.tensor)  # P2-P6
        if self.is_freemask:
            return [features, ]

        if "instances" in batched_inputs[0]:  # 用来处理生成color sim；检测是否有ins，第一个有即可；后面的都会有
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]  # 2*batch个list，每个list的ins数不同
            original_image_masks = [torch.ones_like(x[0], dtype=torch.float32) for x in original_images]
            # mask out the bottom area where the COCO dataset probably has wrong annotations
            for i in range(len(original_image_masks)):  # 2*batch
                im_h = batched_inputs[i]["height"]  # 原图尺寸（未经过数据增强、resize等）
                pixels_removed = int(
                    self.bottom_pixels_removed *
                    float(original_images[i].size(1)) / float(im_h)
                )
                if pixels_removed > 0:
                    original_image_masks[i][-pixels_removed:, :] = 0  # ?? 底部标注不好，去除扰动

            original_images = ImageList.from_tensors(original_images, self.backbone.size_divisibility)
            original_image_masks = ImageList.from_tensors(
                original_image_masks, self.backbone.size_divisibility, pad_value=0.0
            )  # 标准化image_mask（表示何处为真实图像，何处为padding的0）

            # self.add_bitmasks_from_boxes(
            #     gt_instances, original_images.tensor, original_image_masks.tensor,
            #     original_images.tensor.size(-2), original_images.tensor.size(-1)
            # )  # (gt_ins, ori_img, ori_mask,h,w)-->用来处理生成color sim

            if self.use_depth and self.training:
                self.add_bitmasks_from_boxes(
                    gt_instances, original_images.tensor, original_image_masks.tensor,
                    original_images.tensor.size(-2), original_images.tensor.size(-1), depth_prediction=depth_prediction
                )
            else:
                self.add_bitmasks_from_boxes(
                    gt_instances, original_images.tensor, original_image_masks.tensor,
                    original_images.tensor.size(-2), original_images.tensor.size(-1),
                    depth_prediction=None
                )
        else:
            gt_instances = None

        # ins branch
        ins_features = [features[f] for f in self.instance_in_features]
        ins_features = self.split_feats(ins_features)  # P2下采样二倍，P3,3,5不变，P6上采样与P5相同
        cate_pred, kernel_pred, emb_pred = self.ins_head(ins_features)  # SOLOv2InsHead

        # mask branch
        mask_features = [features[f] for f in self.mask_in_features]
        mask_pred = self.mask_head(mask_features)
        # now we got:

        # ins_head
        # kernel_pred(256,40,40), (256,36,36), (256,24,24),(256,16,16), (256,12,12) 有坐标信息coord
        # cate_pred(2,40,40), (2,36,36), (2,24,24),(2,16,16), (2,12,12)  # 预测前景背景， 无坐标信息coord
        # emb_pred(128,40,40), (128,36,36), (128,24,24),(128,16,16), (128,12,12)  没使用

        # mask_head
        # mask_pred(256, h,w) 融合了P2-P5的特征，只有P5融合了coord，hw为规范的尺寸

        if not self.training:
            # point nms.
            cate_pred = [point_nms(cate_p.sigmoid(), kernel=2).permute(0, 2, 3, 1)
                         for cate_p in cate_pred]
            # do inference for results.
            results = self.inference(cate_pred, kernel_pred, emb_pred, mask_pred, images.image_sizes, batched_inputs)
            return results

        elif branch == "supervised":
            mask_feat_size = mask_pred.size()[-2:]
            targets = self.get_ground_truth(gt_instances, mask_feat_size)  # gt尺寸并不规范，输入规范尺存
            losses = self.loss(cate_pred, kernel_pred, emb_pred, mask_pred, targets, depth_prediction=depth_prediction)
            return losses

    def add_bitmasks_from_boxes(self, instances, images, image_masks, im_h, im_w, depth_prediction=None):
        stride = 4
        start = int(stride // 2)

        assert images.size(2) % stride == 0
        assert images.size(3) % stride == 0

        downsampled_images = F.avg_pool2d(
            images.float(), kernel_size=stride,
            stride=stride, padding=0
        )   # 平均池化下采样4倍
        image_masks = image_masks[:, start::stride, start::stride]  # mask（图像部分为1，padding部分为0，remove部分为0）下采样4倍

        for im_i, per_im_gt_inst in enumerate(instances):
            images_lab = color.rgb2lab(downsampled_images[im_i].byte().permute(1, 2, 0).cpu().numpy())  # 图像色彩空间变换
            images_lab = torch.as_tensor(images_lab, device=downsampled_images.device, dtype=torch.float32)
            images_lab = images_lab.permute(2, 0, 1)[None]
            images_color_similarity = get_images_color_similarity(
                images_lab, image_masks[im_i],  # 第i张图，及01mask
                self.pairwise_size, self.pairwise_dilation
            )  # BoxInst

            if depth_prediction is not None:
                image_depth_similarity = get_images_color_similarity(
                    depth_prediction[im_i].unsqueeze(0), image_masks[im_i],
                    self.pairwise_size, self.pairwise_dilation
                )
            per_im_boxes = per_im_gt_inst.gt_boxes.tensor
            per_im_bitmasks = []
            per_im_bitmasks_full = []
            h, w = per_im_gt_inst.image_size
            # per_im_gt_inst.gt_masks = torch.stack(per_im_bitmasks_full, dim=0)[:, :h, :w]
            if len(per_im_gt_inst) > 0:
                per_im_gt_inst.image_color_similarity = torch.cat([
                    images_color_similarity for _ in range(len(per_im_gt_inst))
                ], dim=0)  # 在ins属性中增加色彩相似度，与ins数量无关，一张图像中每个实例的color sim相同

                if depth_prediction is not None:
                    per_im_gt_inst.image_depth_similarity = torch.cat([
                        image_depth_similarity for _ in range(len(per_im_gt_inst))
                    ], dim=0)
