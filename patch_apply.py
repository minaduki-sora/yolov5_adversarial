import os
import os.path as osp
import glob
import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from easydict import EasyDict as edict
import json


IMG_EXT = {".jpg", ".png"}

class PatchApplier:
    def __init__(
            self,
            image_dir: str,
            label_dir: str,
            patch_dir: str,
            class_id: int,
            odd: float,
            cfg: edict,
            train: bool = True
    ):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.patch_dir = patch_dir
        self.class_id = class_id
        self.odd = odd
        self.model_in_sz = cfg.model_in_sz
        self.cfg = cfg
        self.train = train

        t_size_frac = self.cfg.target_size_frac
        mul_gau_mean = self.cfg.mul_gau_mean
        mul_gau_std = self.cfg.mul_gau_std
        x_off_loc = self.cfg.x_off_loc
        y_off_loc = self.cfg.y_off_loc

        self.t_size_frac = [t_size_frac, t_size_frac] if isinstance(t_size_frac, float) else t_size_frac
        self.m_gau_mean = [mul_gau_mean, mul_gau_mean] if isinstance(mul_gau_mean, float) else mul_gau_mean
        self.m_gau_std = [mul_gau_std, mul_gau_std] if isinstance(mul_gau_std, float) else mul_gau_std
        assert (
            len(self.t_size_frac) == 2 and len(self.m_gau_mean) == 2 and len(self.m_gau_std) == 2
        ), "Range must have 2 values"
        self.x_off_loc = x_off_loc
        self.y_off_loc = y_off_loc
        self.min_contrast = 0.8
        self.max_contrast = 1.2
        self.min_brightness = -25
        self.max_brightness = 25
        self.minangle = -20
        self.maxangle = 20
        
    def transform(self, patchs: list[cv2.typing.MatLike]):
        contrast = np.random.uniform(self.min_contrast, self.max_contrast)
        brightness = np.random.uniform(self.min_brightness, self.max_brightness)
        noise = np.random.uniform(-5, 5, patchs[0].shape)
        # add gaussian noise to reduce contrast with a stohastic process
        temp = []
        for i in range(len(patchs)):
            p_c, p_h, p_w = patchs[i].shape
            add_gau = np.random.normal(0, 10, (p_h, p_c, p_w)) if self.train else 0
            adv_patch = patchs[i] + add_gau

            # Apply contrast/brightness/noise
            adv_patch = adv_patch * contrast + brightness + noise if self.train else adv_patch
            # Clamp values
            temp.insert(i, np.clip(adv_patch, 0, 255))
            # cv2.imwrite(f"test/out/patch_{i}.png", patchs[i])
        return temp

    def rotate(self, patch: cv2.typing.MatLike, angle: int):
        height = patch.shape[0]
        width = patch.shape[1]
        new_height = int(width * np.abs(np.sin(angle)) + height * np.abs(np.cos(angle)))
        new_width = int(height * np.abs(np.sin(angle)) + width * np.abs(np.cos(angle)))
        # rotate the patch
        M = cv2.getRotationMatrix2D((width//2, height//2), angle, 1)
        new_center = ((new_width - 1) / 2.0, (new_height - 1) / 2.0)
        M[0, 2] += (new_center[0] - width / 2.0)
        M[1, 2] += (new_center[1] - height / 2.0)
        patch = cv2.warpAffine(patch, M, (new_width, new_height))
        return patch

    def apply(self):
        if not all([osp.isdir(self.image_dir), osp.isdir(self.label_dir)]):
            raise ValueError(f"image_dir, label_dir and patch_dir must be directories")
        src_image_path = osp.join(self.image_dir, "*")
        src_label_path = osp.join(self.label_dir, "*")
        src_label_paths = sorted(glob.glob(src_label_path))
        src_image_paths = [p for p in sorted(glob.glob(src_image_path)) if osp.splitext(p)[-1] in IMG_EXT]
        assert len(src_image_paths) == len(src_label_paths)

        patch_paths = osp.join(self.patch_dir, "*.png")
        patchs = [cv2.imread(p) for p in sorted(glob.glob(patch_paths))]
        with tqdm.tqdm(total=len(src_image_path)) as pbar:
            for img_path, lab_path in zip(src_image_paths, src_label_paths):
                img = cv2.imread(img_path)
                with open(lab_path, 'r') as lab:
                    patch_ = self.transform(patchs)
                    additional_patch_labels = []
                    for label in lab:
                        # Add patch with a probability of odd
                        if np.random.rand() > self.odd:
                            continue
                        label = label.strip().split()
                        x, y, w, h = map(float, label[1:])
                        # select a random patch
                        patch = patch_[np.random.randint(0, len(patch_))]
                        # resize the patch to 30% size of the bounding box
                        ratio = math.sqrt(w * img.shape[1] * img.shape[0] * h * 0.3 / (patch.shape[0] * patch.shape[1]))
                        if ratio < 0.016:
                            continue
                        patch = cv2.resize(patch, (int(ratio * patch.shape[1]), int(ratio * patch.shape[0])))
                        # select a random angle to rotate the patch
                        angle = np.random.randint(self.minangle, self.maxangle)
                        patch = self.rotate(patch, angle)
                        # attach the patch to the image
                        x1 = int((x - w/2) * img.shape[1])
                        y1 = int((y - h/2) * img.shape[0])
                        x2 = int((x + w/2) * img.shape[1])
                        y2 = int((y + h/2) * img.shape[0])
                        
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2

                        for i in range(patch.shape[0]):
                            for j in range(patch.shape[1]):
                                x = center_x - patch.shape[1] // 2 + j
                                y = center_y - patch.shape[0] // 2 + i
                                if x >= 0 and x < img.shape[1] and y >= 0 and y < img.shape[0] and patch[i, j].sum() > 0:
                                    img[y, x] = patch[i, j]
                        additional_patch_labels.append([self.class_id, center_x / img.shape[1], center_y / img.shape[0], patch.shape[1] / img.shape[1], patch.shape[0] / img.shape[0]])

                cv2.imwrite(img_path, img)
                with open(lab_path, 'a') as f:
                    for label in additional_patch_labels:
                        f.write(" ".join(map(str, label)))
                        f.write("\n")
                
                pbar.update(1)

def load_config_object(cfg_path: str) -> edict:
    """Loads a config json and returns a edict object."""
    with open(cfg_path, "r", encoding="utf-8") as json_file:
        cfg_dict = json.load(json_file)

    return edict(cfg_dict)


if __name__ == "__main__":
    pa = PatchApplier(
        image_dir="../datasets/VisDrone_patch/VisDrone2019-DET-test-dev/images",
        label_dir="../datasets/VisDrone_patch/VisDrone2019-DET-test-dev/labels",
        patch_dir="runs/compare/patch",
        class_id=4,
        odd=1,
        cfg=load_config_object("test/cfg.json"),
        train=False
    )
    pa.apply()
        