import os
from turtle import right

import numpy as np
from PIL import Image
from enum import IntEnum


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

