# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/FreeSOLO/blob/main/LICENSE

# -------------------------------------------------------------------------
# MIT License
#
# Copyright (c) 2021 Facebook, Inc. and its affiliates.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Modified by Xinlong Wang
# -------------------------------------------------------------------------
import torch
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch

from freesolo import add_solo_config
from freesolo.engine.my_trainer import My_BaselineTrainer

# hacky way to register
import freesolo.data.datasets.builtin
from freesolo.modeling.solov2 import My_PseudoSOLOv2
from freesolo import my_eval_cocoapi


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()  # detectron2\config\defaults.py
    add_solo_config(cfg)  # MODEL.SOLOV2
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    Trainer = My_BaselineTrainer
    # print('\n******************************')

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
            # resume_or_lord(path, whether_resume)
        )
        res = Trainer.test(cfg, model)
        my_eval_cocoapi.eval_my_coco(cfg)
        return res

    trainer = Trainer(cfg)
    # trainer.checkpointer.save("last_epoch_ckpt", )
    trainer.resume_or_load(resume=args.resume)  # false

    return trainer.train()
    # 在每轮训练当中，Trainer 都会建立一个EventStorage 的对象 self.storage ， 并且通过一系列的方法，可以使得在run_step
    # 的每个细节中，都可以访问到这个对象，并将数据记录在这个对象当中，并且在after_step方法当中获取记录的这些信息，
    # 用于完成日志的记录和其他相关的操作。可以简单的认为EventStorage 就是 run_step 和after_step 之间的通信方式


if __name__ == "__main__":
    args = default_argument_parser().parse_args()

    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
