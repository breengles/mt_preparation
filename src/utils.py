import os

import numpy as np
from PIL import Image
from enum import IntEnum
from tqdm.auto import tqdm
import shutil
from .faceparsing import evaluate


class Regions(IntEnum):
    """
    commented -- original SCGAN labels
    used -- labels from faceparsing
    """

    BACKGROUND = 0  # 0
    FACE = 1  # 4
    LEFT_EYEBROW = 2  # 7
    RIGHT_EYEBROW = 3  # 2
    LEFT_EYE = 4  # 6
    RIGHT_EYE = 5  # 1
    NOSE = 10  # 8
    TEETH = 11  # 11
    UPPER_LIP_VERMILLION = 12  # 9
    LOWER_LIP_VERMILLION = 13  # 13
    NECK = 14  # 10
    HAIR = 17  # 12


def change_labels(seg):
    """
    To match SCGAN segmentation
    """
    new = np.zeros_like(seg)
    new[seg == Regions.BACKGROUND.value] = 0
    new[seg == Regions.FACE.value] = 4
    new[seg == Regions.LEFT_EYEBROW.value] = 7
    new[seg == Regions.RIGHT_EYEBROW.value] = 2
    new[seg == Regions.LEFT_EYE.value] = 6
    new[seg == Regions.RIGHT_EYE.value] = 1
    new[seg == 6] = 0
    new[seg == 7] = 0
    new[seg == 8] = 0
    new[seg == 9] = 0
    new[seg == Regions.NOSE.value] = 8
    new[seg == Regions.TEETH.value] = 11
    new[seg == Regions.UPPER_LIP_VERMILLION.value] = 9
    new[seg == Regions.LOWER_LIP_VERMILLION.value] = 13
    new[seg == Regions.NECK.value] = 10
    new[seg == 15] = 0
    new[seg == 16] = 0
    new[seg == Regions.HAIR.value] = 12
    return new


def generate_dataset_txt(dataroot, kind):
    img_dir = os.path.join(dataroot, "images", kind)

    dataset_txt = os.path.join(dataroot, f"{kind}.txt")

    with open(dataset_txt, "w+") as f:
        for img_name in os.listdir(img_dir):
            f.write(img_name + "\n")


def invalid_segmentation(seg):
    """
    original labels from faceparsing are used
    """
    skin = (seg == Regions.FACE.value).sum() == 0
    lip_top = (seg == Regions.UPPER_LIP_VERMILLION.value).sum() == 0
    lip_bot = (seg == Regions.LOWER_LIP_VERMILLION.value).sum() == 0
    nose = (seg == Regions.NOSE.value).sum() == 0
    eye_right = (seg == Regions.RIGHT_EYE.value).sum() == 0
    eyebrow_right = (seg == Regions.RIGHT_EYEBROW.value).sum() == 0
    eye_left = (seg == Regions.LEFT_EYE.value).sum() == 0
    eyebrow_left = (seg == Regions.LEFT_EYEBROW.value).sum() == 0

    return skin or lip_top or lip_bot or nose or eye_right or eyebrow_right or eye_left or eyebrow_left


def cut_background(img, seg):
    if img.size != seg.size:
        seg = seg.resize(img.size, Image.NEAREST)

    seg_ = np.array(seg)
    out = np.array(img).copy()
    out[seg_ == Regions.BACKGROUND.value] = 0
    return out


def crop_face(img, seg):
    if img.size != seg.size:
        seg = seg.resize(img.size, Image.NEAREST)

    seg_ = np.array(seg)

    y_index, x_index = np.nonzero(seg_ == Regions.FACE.value)  # I hate numpy...

    out_img = img.crop((min(x_index), min(y_index), max(x_index), max(y_index)))
    out_seg = seg.crop((min(x_index), min(y_index), max(x_index), max(y_index)))

    return out_img, out_seg


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

    parsings = evaluate(dspth=datapath, cp="res/cp/79999_iter.pth", respth=testdir, desc=f"Parsing {dataroot}")

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
