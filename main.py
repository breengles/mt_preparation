#!/usr/bin/env python


import pyrallis

from src import TaskConfig, segment_dataset, cut_dataset, crop_dataset, cutcrop_dataset, Aligner, align_dataset
from dataclasses import asdict


@pyrallis.wrap()
def main(cfg: TaskConfig):
    dataroot = cfg.dataroot

    assert dataroot != "", "provide dataroot"

    if cfg.align:
        aligner = Aligner(**asdict(cfg.align_config))
        align_dataset(cfg.dataroot, aligner)
        dataroot = cfg.dataroot + ".align"

    segment_dataset(dataroot, scgan_labels=cfg.scgan_labels)

    if cfg.cut and cfg.crop:
        cutcrop_dataset(dataroot)
    elif cfg.cut and not cfg.crop:
        cut_dataset(dataroot)
    elif not cfg.cut and cfg.crop:
        crop_dataset(dataroot)


if __name__ == "__main__":
    main()
