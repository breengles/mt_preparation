#!/usr/bin/env python

from configparser import Interpolation
import os
import shutil
from argparse import ArgumentParser
from uuid import uuid4
from PIL import Image
from tqdm.auto import tqdm


def resize_dataset(dataroot, kind, size=512):
    images_dir = os.path.join(dataroot, "images", kind)
    seg_dir = os.path.join(dataroot, "segments", kind)

    for img_name in tqdm(os.listdir(images_dir), desc=f"Resizing {kind}"):
        img = Image.open(os.path.join(images_dir, img_name))
        seg = Image.open(os.path.join(seg_dir, img_name))

        img = img.resize((size, size))
        seg = seg.resize((size, size), resample=Image.NEAREST)

        img.save(os.path.join(images_dir, img_name))
        seg.save(os.path.join(seg_dir, img_name))


def reindex_dataset(dataset_root, kind):
    img_root = os.path.join(dataset_root, "images")
    seg_root = os.path.join(dataset_root, "segments")

    img_dir = os.path.join(img_root, kind)
    seg_dir = os.path.join(seg_root, kind)

    for idx, img_name in enumerate(os.listdir(img_dir)):
        img_path = os.path.join(img_dir, img_name)
        img_ext = img_path.split(".")[-1]  # strong assumption yeep
        img_new_path = os.path.join(img_dir, f"{idx}.{img_ext}")
        os.rename(img_path, img_new_path)

        seg_path = os.path.join(seg_dir, img_name)
        seg_ext = seg_path.split(".")[-1]
        seg_new_path = os.path.join(seg_dir, f"{idx}.{seg_ext}")
        os.rename(seg_path, seg_new_path)


def get_dataset_name(dataset_roots):
    names = []
    for dataset_root in dataset_roots:
        names.append(os.path.basename(dataset_root))

    return ".".join(names)


def copy_files(src_root, dst_root, kind):
    img_names = os.listdir(os.path.join(src_root, "images"))

    for img_name in img_names:
        img_path = os.path.join(src_root, "images", img_name)
        seg_path = os.path.join(src_root, "segments", img_name)

        dst_name = str(uuid4()) + ".png"
        dst_img_path = os.path.join(dst_root, "images", kind, dst_name)
        dst_seg_path = os.path.join(dst_root, "segments", kind, dst_name)

        shutil.copy(img_path, dst_img_path)
        shutil.copy(seg_path, dst_seg_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--makeup_roots", nargs="+")
    parser.add_argument("--nonmakeup_roots", nargs="+")
    parser.add_argument("--savedir", default="datasets", type=str)
    parser.add_argument("--resize", type=int, default=None)

    args = parser.parse_args()

    makeup_dataset_name = get_dataset_name(args.makeup_roots)
    nonmakeup_dataset_name = get_dataset_name(args.nonmakeup_roots)

    if makeup_dataset_name == nonmakeup_dataset_name:
        dataset_name = f"MT-{makeup_dataset_name}"
    else:
        dataset_name = f"MT-{makeup_dataset_name}_{nonmakeup_dataset_name}"

    if args.resize is not None:
        dataset_name = dataset_name + f"_{args.resize}px"

    dataset_root = os.path.join(args.savedir, dataset_name)
    images_path = os.path.join(dataset_root, "images")
    segments_path = os.path.join(dataset_root, "segments")

    os.makedirs(dataset_root, exist_ok=True)
    os.makedirs(images_path, exist_ok=True)
    os.makedirs(os.path.join(images_path, "makeup"), exist_ok=True)
    os.makedirs(os.path.join(images_path, "non-makeup"), exist_ok=True)
    os.makedirs(segments_path, exist_ok=True)
    os.makedirs(os.path.join(segments_path, "makeup"), exist_ok=True)
    os.makedirs(os.path.join(segments_path, "non-makeup"), exist_ok=True)

    for makeup_root in args.makeup_roots:
        copy_files(makeup_root, dataset_root, "makeup")

    for nonmakeup_root in args.nonmakeup_roots:
        copy_files(nonmakeup_root, dataset_root, "non-makeup")

    reindex_dataset(dataset_root, "makeup")
    reindex_dataset(dataset_root, "non-makeup")

    if args.resize is not None:
        resize_dataset(dataset_root, "makeup", size=args.resize)
        resize_dataset(dataset_root, "non-makeup", size=args.resize)
