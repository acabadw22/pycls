#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Compute precise time for a model on 1 GPU."""

import os

import pycls.core.config as config
import pycls.core.distributed as dist
import pycls.core.trainer as trainer
from pycls.core.config import cfg


def main():
    config.load_cfg_fom_args("Compute precise time for a model on 1 GPU.")
    # Currently the code is only tested for the GPU=1 case
    cfg.NUM_GPUS = 1
    cfg.PREC_TIME.ENABLED = True
    config.assert_and_infer_cfg()
    cfg.freeze()
    os.makedirs(cfg.OUT_DIR, exist_ok=True)
    dist.multi_proc_run(num_proc=cfg.NUM_GPUS, fun=trainer.time_model)


if __name__ == "__main__":
    main()
