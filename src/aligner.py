import os

import dlib
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
from scipy import ndimage
import shutil


class Aligner:
    """
    see https://github.com/yuval-alaluf/stylegan3-editing
    """

    def __init__(
        self,
        shape_predictor_path="shape_predictor_68_face_landmarks.dat",
        transform_size=1024,
        padding=True,
        output_size=None,
    ) -> None:
        self.predictor = dlib.shape_predictor(shape_predictor_path)
        self.detector = dlib.get_frontal_face_detector()

        self.transform_size = transform_size
        self.enable_padding = padding
        self.output_size = output_size if output_size is not None else transform_size

    @staticmethod
    def _get_eyes_coords(landmarks):
        lm_eye_left = landmarks[36:42]  # left-clockwise
        lm_eye_right = landmarks[42:48]  # left-clockwise

        # Calculate auxiliary vectors.
        eye_left = np.mean(lm_eye_left, axis=0)
        eye_right = np.mean(lm_eye_right, axis=0)

        return eye_left, eye_right

    def _get_landmark(self, img_path):
        img = dlib.load_rgb_image(img_path)

        shape = None
        for d in self.detector(img, 1):
            shape = self.predictor(img, d)

        if shape is None:
            return None

        t = list(shape.parts())
        a = []
        for tt in t:
            a.append([tt.x, tt.y])

        return np.array(a)

    def _get_positions(self, img_path):
        lm = self._get_landmark(img_path)

        if lm is None:
            return None, None, None

        lm_mouth_outer = lm[48:60]  # left-clockwise

        # Calculate auxiliary vectors.
        eye_left, eye_right = self._get_eyes_coords(lm)
        eye_avg = (eye_left + eye_right) * 0.5
        eye_to_eye = eye_right - eye_left
        mouth_left = lm_mouth_outer[0]
        mouth_right = lm_mouth_outer[6]
        mouth_avg = (mouth_left + mouth_right) * 0.5
        eye_to_mouth = mouth_avg - eye_avg

        # Choose oriented crop rectangle.
        x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
        x /= np.hypot(*x)
        x *= np.hypot(*eye_to_eye) * 2.0
        y = np.flipud(x) * [-1, 1]
        c = eye_avg + eye_to_mouth * 0.1

        return c, x, y

    def _get_transforms(self, c, x, y):
        quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
        qsize = np.hypot(*x) * 2
        return quad, qsize

    def _transform(self, img_path: str, quad: np.ndarray, qsize: int):
        # read image
        img = Image.open(img_path)

        # Shrink.
        shrink = int(np.floor(qsize / self.output_size * 0.5))
        if shrink > 1:
            rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
            img = img.resize(rsize, Image.ANTIALIAS)
            quad /= shrink
            qsize /= shrink

        # Crop.
        border = max(int(np.rint(qsize * 0.1)), 3)
        crop = (
            int(np.floor(min(quad[:, 0]))),
            int(np.floor(min(quad[:, 1]))),
            int(np.ceil(max(quad[:, 0]))),
            int(np.ceil(max(quad[:, 1]))),
        )
        crop = (
            max(crop[0] - border, 0),
            max(crop[1] - border, 0),
            min(crop[2] + border, img.size[0]),
            min(crop[3] + border, img.size[1]),
        )
        if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
            img = img.crop(crop)
            quad -= crop[0:2]

        # Pad.
        pad = (
            int(np.floor(min(quad[:, 0]))),
            int(np.floor(min(quad[:, 1]))),
            int(np.ceil(max(quad[:, 0]))),
            int(np.ceil(max(quad[:, 1]))),
        )
        pad = (
            max(-pad[0] + border, 0),
            max(-pad[1] + border, 0),
            max(pad[2] - img.size[0] + border, 0),
            max(pad[3] - img.size[1] + border, 0),
        )
        if self.enable_padding and max(pad) > border - 4:
            pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
            img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), "reflect")
            h, w, _ = img.shape
            y, x, _ = np.ogrid[:h, :w, :1]
            mask = np.maximum(
                1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]),
                1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3]),
            )
            blur = qsize * 0.02
            img += (ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
            img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
            img = Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), "RGB")
            quad += pad[:2]

        # Transform.
        img = img.transform(
            (self.transform_size, self.transform_size), Image.QUAD, (quad + 0.5).flatten(), Image.BILINEAR
        )
        if self.output_size < self.transform_size:
            img = img.resize((self.output_size, self.output_size), Image.ANTIALIAS)

        # Return aligned image.
        return img

    def align(self, img_path):
        c, x, y = self._get_positions(img_path)

        if c is None and x is None and y is None:
            return None

        quad, qsize = self._get_transforms(c, x, y)
        img = self._transform(img_path, quad, qsize)
        return img


def align_dataset(dataroot: str, aligner: Aligner):
    datapath = os.path.join(dataroot, "images")
    aligned_root = dataroot + ".align"

    savepath = os.path.join(aligned_root, "images")
    trashpath = os.path.join(aligned_root, "trash")

    os.makedirs(savepath, exist_ok=True)
    os.makedirs(trashpath, exist_ok=True)

    for img_name in tqdm(os.listdir(datapath), desc=f"Aligning {datapath}"):
        img_path = os.path.join(datapath, img_name)

        aligned_img = aligner.align(img_path)

        if aligned_img is None:
            shutil.copy(img_path, os.path.join(trashpath, img_name))
        else:
            aligned_img.save(os.path.join(savepath, img_name))

    print(f"Trashed {len(os.listdir(trashpath))}/{len(os.listdir(datapath))}")
