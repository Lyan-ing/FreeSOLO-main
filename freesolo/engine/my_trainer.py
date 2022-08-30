# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/FreeSOLO/blob/main/LICENSE

import os
import copy
import time
import logging
import torch
from torch.nn.parallel import DistributedDataParallel
import torch.nn.functional as F
from fvcore.nn.precise_bn import get_bn_modules
import numpy as np

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import DefaultTrainer, SimpleTrainer, TrainerBase
# from detectron2.engine.train_loop import AMPTrainer
from detectron2.utils.events import EventStorage
from detectron2.evaluation import COCOEvaluator, verify_results
from detectron2.engine import hooks
from detectron2.structures.instances import Instances
from detectron2.utils.env import TORCH_VERSION

from freesolo.data.build import (
    build_detection_semisup_train_loader_two_crops,
)
from freesolo.data.my_dataset_mapper import DatasetMapperTwoCropSeparate


class My_BaselineTrainer(DefaultTrainer):
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        Use the custom checkpointer, which loads other backbone models
        with matching heuristics.
        """
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)

        if comm.get_world_size() > 1:
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], broadcast_buffers=False, find_unused_parameters=True
            )

        TrainerBase.__init__(self)
        self._trainer = SimpleTrainer(
            model, data_loader, optimizer
        )  # init trainer

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)  # init lr_scheduler
        self.checkpointer = DetectionCheckpointer(
            model,
            cfg.OUTPUT_DIR,
            optimizer=optimizer,
            scheduler=self.scheduler,
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())  # how to use the hook? and how to register?

    def resume_or_load(self, resume=True):
        """
            If `resume==True` and `cfg.OUTPUT_DIR` contains the last checkpoint (defined by
            a `last_checkpoint` file), resume from the file. Resuming means loading all
            available states (eg. optimizer and scheduler) and update iteration counter
            from the checkpoint. ``cfg.MODEL.WEIGHTS`` will not be used.
            Otherwise, this is considered as an independent training. The method will load model
            weights from the file `cfg.MODEL.WEIGHTS` (but will not load other states) and start
            from iteration 0.
            Args:
                resume (bool): whether to do resume or not
            """
        checkpoint = self.checkpointer.resume_or_load(
            self.cfg.MODEL.WEIGHTS, resume=resume
        )
        if resume and self.checkpointer.has_checkpoint():
            self.start_iter = checkpoint.get("iteration", -1) + 1
            # The checkpoint stores the training iteration that just finished, thus we start
            # at the next iteration (or iter zero if there's no checkpoint).
        if isinstance(self.model, DistributedDataParallel):
            # broadcast loaded data/model from the first rank, because other
            # machines may not have access to the checkpoint file
            if TORCH_VERSION >= (1, 7):
                self.model._sync_params_and_buffers()
            self.start_iter = comm.all_gather(self.start_iter)[0]

    def copy_and_paste(self, labeled_data, unlabeled_data):
        new_unlabeled_data = []  # data_k(-1), data_q

        def mask_iou_matrix(x, y, mode='iou'):  # 图像大小相同的实例mask，ins数量不同
            x = x.reshape(x.shape[0], -1).float()  # cpoied
            y = y.reshape(y.shape[0], -1).float()  # unlabled
            inter_matrix = x @ y.transpose(1, 0)  # n1xn2
            sum_x = x.sum(1)[:, None].expand(x.shape[0], y.shape[0])
            sum_y = y.sum(1)[None, :].expand(x.shape[0], y.shape[0])
            if mode == 'ioy':
                iou_matrix = inter_matrix / (sum_y)
            else:
                iou_matrix = inter_matrix / (sum_x + sum_y - inter_matrix)
            return iou_matrix

        def visualize_data(data, cfg, save_path='./sample.jpg'):
            from detectron2.data import detection_utils as utils
            from detectron2.data import DatasetCatalog, MetadataCatalog
            from detectron2.utils.visualizer import Visualizer
            data["instances"] = data["instances"].to(device='cpu')
            img = data["image"].permute(1, 2, 0).cpu().detach().numpy()
            img = utils.convert_image_to_rgb(img, cfg.INPUT.FORMAT)
            metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN_LABEL[0])
            visualizer = Visualizer(img, metadata=metadata, scale=1.0)
            target_fields = data["instances"].get_fields()
            labels = [metadata.thing_classes[i] for i in target_fields["gt_classes"]]
            vis = visualizer.overlay_instances(
                labels=labels,
                boxes=target_fields.get("gt_boxes", None),
                masks=target_fields.get("gt_masks", None),
                keypoints=target_fields.get("gt_keypoints", None),
            )
            print("Saving to {} ...".format(save_path))
            vis.save(save_path)

        def convert_dimension(imgs):
            img_h = int(imgs.shape[1] / 2)
            ori_img = imgs[:, :img_h]
            depth_map = imgs[:, img_h:]
            re_img = torch.cat((ori_img, depth_map), axis=0)
            return re_img

        def de_convert_dimension(imgs):
            ori_img = imgs[:3, ]
            depth_map = imgs[3:, ]
            re_img = torch.cat((ori_img, depth_map), axis=1)
            return re_img

        for cur_labeled_data, cur_unlabeled_data in zip(labeled_data, unlabeled_data):  # 一张张读取
            cur_labeled_instances = cur_labeled_data["instances"]
            cur_labeled_image = cur_labeled_data["image"]
            cur_unlabeled_instances = cur_unlabeled_data["instances"]
            cur_unlabeled_image = cur_unlabeled_data["image"]
            if self.cfg.MODEL.SOLOV2.USE_DEPTH:
                cur_labeled_image = convert_dimension(cur_labeled_image)
                cur_unlabeled_image = convert_dimension(cur_unlabeled_image)

            num_labeled_instances = len(cur_labeled_instances)
            num_copy = np.random.randint(max(1, num_labeled_instances + 1))  # 随机取0，1...num_labeled_ins个
            if num_labeled_instances == 0 or num_copy == 0:  # 如果label没有实例或复制0次label实例，
                new_unlabeled_data.append(cur_unlabeled_data)  # 就直接添加当前unlabel数据（包含图像和实例mask）
            else:  # label有实例或复制多次label的实例时：
                choice = np.random.choice(num_labeled_instances, num_copy, replace=False)
                copied_instances = cur_labeled_instances[choice].to(device=cur_unlabeled_instances.gt_boxes.device)
                copied_masks = copied_instances.gt_masks
                copied_boxes = copied_instances.gt_boxes
                _, labeled_h, labeled_w = cur_labeled_image.shape
                _, unlabeled_h, unlabeled_w = cur_unlabeled_image.shape
                # rescale the labeled image to align with unlabeled one.
                cur_labeled_image = F.interpolate(cur_labeled_image[None, ...].float(), size=(unlabeled_h, unlabeled_w),
                                                  mode="bilinear", align_corners=False).byte().squeeze(0)
                copied_masks.tensor = F.interpolate(copied_masks.tensor[None, ...].float(),
                                                    size=(unlabeled_h, unlabeled_w), mode="bilinear",
                                                    align_corners=False).bool().squeeze(0)
                copied_boxes.scale(1. * unlabeled_w / labeled_w, 1. * unlabeled_h / labeled_h)
                #
                copied_instances.gt_masks = copied_masks
                copied_instances.gt_boxes = copied_boxes
                copied_instances._image_size = (unlabeled_h, unlabeled_w)
                if len(cur_unlabeled_instances) == 0:
                    alpha = copied_instances.gt_masks.tensor.sum(0) > 0
                    # merge image
                    alpha = alpha.cpu()
                    composited_image = (alpha * cur_labeled_image) + (~alpha * cur_unlabeled_image)
                    if self.cfg.MODEL.SOLOV2.USE_DEPTH:
                        composited_image = de_convert_dimension(composited_image)
                    cur_unlabeled_data["image"] = composited_image
                    cur_unlabeled_data["instances"] = copied_instances
                else:
                    # remove the copied object if iou greater than 0.5
                    iou_matrix = mask_iou_matrix(copied_masks.tensor, cur_unlabeled_instances.gt_masks.tensor,
                                                 mode='ioy')  # nxN
                    # print(iou_matrix)
                    keep = iou_matrix.max(1)[0] < 0.5  # [0]表示最大值，[1]表示最大值的索引
                    if keep.sum() == 0:  # 如果全为false，即k图中每个实例与q的每个实例最大iou都大于0.5，就不mix
                        new_unlabeled_data.append(cur_unlabeled_data)  # 原图，不经过mix
                        continue
                    copied_instances = copied_instances[keep]
                    # update existing instances in unlabeled image
                    alpha = copied_instances.gt_masks.tensor.sum(0) > 0  # 0-->false
                    cur_unlabeled_instances.gt_masks.tensor = ~alpha * cur_unlabeled_instances.gt_masks.tensor
                    # merge image
                    alpha = alpha.cpu()
                    composited_image = (alpha * cur_labeled_image) + (~alpha * cur_unlabeled_image)
                    # merge instances, 将k图中有实例的部分贴进q中无实例的部分
                    merged_instances = Instances.cat([cur_unlabeled_instances, copied_instances])
                    # update boxes
                    merged_instances.gt_boxes = merged_instances.gt_masks.get_bounding_boxes()
                    if self.cfg.MODEL.SOLOV2.USE_DEPTH:
                        composited_image = de_convert_dimension(composited_image)
                    cur_unlabeled_data["image"] = composited_image
                    cur_unlabeled_data["instances"] = merged_instances
                    import torchshow
                    torchshow.save(composited_image,"./conbine.png")
                # visualize_data(cur_unlabeled_data, self.cfg, save_path = './sample_{}.jpg'.format(np.random.randint(10)))
                new_unlabeled_data.append(cur_unlabeled_data)
        return new_unlabeled_data

    def run_step(self):
        self._trainer.iter = self.iter

        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        # if comm.is_main_process():
        #     # self.model.
        #     self.checkpointer.save("last_epoch_ckpt")

        data = next(self._trainer._data_loader_iter)
        data_q, data_k = data
        data_time = time.perf_counter() - start
        data_q = self.copy_and_paste(copy.deepcopy(data_k[::-1]), data_q)
        # (复制data_k并将图像反序，data_k不变；再对k,进行copy_and_paste：数据增强的方式，将两张图的实例在空白处贴合-->深度估计不能用)
        data_q.extend(data_k)  # 数据增强后的放一起，获得了两倍的数据量
        record_dict = self.model(data_q, branch="supervised")  # images ?

        loss_dict = {}
        for key in record_dict.keys():
            if "loss" in key and key[-3:] != "val":
                loss_dict[key] = record_dict[key]

        losses = sum(loss_dict.values())

        metrics_dict = record_dict
        metrics_dict["data_time"] = data_time
        self._write_metrics(metrics_dict)

        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DatasetMapperTwoCropSeparate(cfg, True)
        # This customized mapper produces two augmented images from a single image
        # instance. This mapper makes sure that the two augmented images have the same
        # cropping and thus the same size.
        return build_detection_semisup_train_loader_two_crops(cfg, mapper)

    def _write_metrics(self, metrics_dict: dict):
        """
        Args:
            metrics_dict (dict): dict of scalar metrics
        """
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        # gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only
        # supported method in detectron2.
        all_metrics_dict = comm.gather(metrics_dict)

        if comm.is_main_process():
            if "data_time" in all_metrics_dict[0]:
                data_time = np.max([x.pop("data_time")
                                    for x in all_metrics_dict])
                self.storage.put_scalar("data_time", data_time)

            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict])
                for k in all_metrics_dict[0].keys()
            }

            loss_dict = {}
            for key in metrics_dict.keys():
                if key[:4] == "loss":
                    loss_dict[key] = metrics_dict[key]

            total_losses_reduced = sum(loss for loss in loss_dict.values())

            self.storage.put_scalar("total_loss", total_losses_reduced)
            if len(metrics_dict) > 1:
                self.storage.put_scalars(**metrics_dict)
