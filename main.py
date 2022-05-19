#!/usr/bin/env python

import os
import shutil
from argparse import ArgumentParser

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from src.faceparsing.test import evaluate
from utils import change_labels, crop_face, cut_background, invalid_segmentation


def cut_dataset(dataroot):
    """
    Remove background from images using segmentation with labels from SCGAN.
    Segs are resized to match image size
    """
    cut_root = dataroot + ".cut"
    cut_images_path = os.path.join(cut_root, "images")
    cut_seg_path = os.path.join(cut_root, "segments")

    images_path = os.path.join(dataroot, "images")
    seg_path = os.path.join(dataroot, "segments")

    os.makedirs(cut_images_path, exist_ok=True)

    for img_name in tqdm(os.listdir(images_path), desc=f"Cutting {dataroot}"):
        img = Image.open(os.path.join(images_path, img_name))
        seg = Image.open(os.path.join(seg_path, img_name))

        cut_img = Image.fromarray(cut_background(img, seg))
        cut_img.save(os.path.join(cut_images_path, img_name))

    shutil.copytree(seg_path, cut_seg_path)

    return cut_root


def crop_dataset(dataroot):
    """
    Crop face from images using segmentation with labels from SCGAN.
    Segs are resized to match image size
    """
    crop_root = dataroot + ".crop"
    crop_images_path = os.path.join(crop_root, "images")
    crop_seg_path = os.path.join(crop_root, "segments")

    images_path = os.path.join(dataroot, "images")
    seg_path = os.path.join(dataroot, "segments")

    os.makedirs(crop_images_path, exist_ok=True)
    os.makedirs(crop_seg_path, exist_ok=True)

    for img_name in tqdm(os.listdir(images_path), desc=f"Cropping {dataroot}"):
        img = Image.open(os.path.join(images_path, img_name))
        seg = Image.open(os.path.join(seg_path, img_name))

        out_img, out_seg = crop_face(img, seg)
        out_img.save(os.path.join(crop_images_path, img_name))
        out_seg.save(os.path.join(crop_seg_path, img_name))

    return crop_root


def segment_dataset(dataroot="makeup/dataset", scgan_labels=False):
    datapath = os.path.join(dataroot, "images")
    savedir = os.path.join(dataroot, "segments")
    testdir = os.path.join(dataroot, "test")
    trashdir = os.path.join(dataroot, "trash")

    os.makedirs(savedir, exist_ok=True)
    os.makedirs(testdir, exist_ok=True)
    os.makedirs(trashdir, exist_ok=True)

    parsings = evaluate(dspth=datapath, cp="79999_iter.pth", respth=testdir, desc=f"Parsing {dataroot}")

    total_trashed = 0
    total_images = 0
    for img_name, seg in parsings:
        total_images += 1

        if invalid_segmentation(seg):
            total_trashed += 1
            shutil.move(os.path.join(datapath, img_name), os.path.join(trashdir, img_name))
            continue

        if scgan_labels:
            seg = change_labels(seg)

        seg_img = Image.fromarray(seg.astype(np.uint8))

        img_size = Image.open(os.path.join(datapath, img_name)).size
        seg_img = seg_img.resize(img_size, Image.NEAREST)

        seg_img.save(os.path.join(savedir, img_name))

    print(f"Images trashed: {total_trashed} / {total_images}")


def cutcrop_dataset(dataset_root):
    cut_dataset_root = cut_dataset(dataset_root)  # remove background from original dataset
    crop_dataset_root = crop_dataset(dataset_root)  # crop original dataset
    cutcrop_dataset_root = crop_dataset(cut_dataset_root)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("dataroot")
    parser.add_argument("--crop", action="store_true")
    parser.add_argument("--cut", action="store_true")
    parser.add_argument("--scgan_labels", action="store_true")

    args = parser.parse_args()

    segment_dataset(args.dataroot, scgan_labels=args.scgan_labels)

    if args.cut and args.crop:
        cutcrop_dataset(args.dataroot)
    elif args.cut and not args.crop:
        cut_dataset(args.dataroot)
    elif not args.cut and args.crop:
        crop_dataset(args.dataroot)
