# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/FreeSOLO/blob/main/LICENSE

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def eval_my_coco(cfg):
    annType = 'segm'
    prefix = 'instances'
    print('Running demo for *{}* results.'.format(annType))

    dataDir = '/ceph-jd/pub/jupyter/yangyang/notebooks/coco/'
    dataType = 'val2017'
    annFile = '{}/annotations/{}_{}.json'.format(dataDir, prefix, dataType)
    cocoGt = COCO(annFile)

    resFile = cfg.OUTPUT_DIR+'/inference/coco_instances_results.json'
    # resFile = 'demo/instances_val2017_densecl_r101.json'
    cocoDt = cocoGt.loadRes(resFile)

    cocoEval = COCOeval(cocoGt, cocoDt, annType)
    cocoEval.params.useCats = 0
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
