"""
Testing code for evaluating Adversarial patches against object detection

python test_patch.py --cfg config_json_file
"""
import io
import os
import os.path as osp
import time
import json
import glob
from pathlib import Path
from typing import Optional
from contextlib import redirect_stdout

import tqdm
import numpy as np
from PIL import Image
from easydict import EasyDict as edict
import torch
from torchvision import transforms
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from models.common import DetectMultiBackend
from utils.metrics import ConfusionMatrix
from utils.general import non_max_suppression, xyxy2xywh
from utils.torch_utils import select_device

from adv_patch_gen.utils.config_parser import get_argparser, load_config_object
from adv_patch_gen.utils.patch import PatchApplier, PatchTransformer


IMG_EXTNS = {".png", ".jpg", ".jpeg"}
CLASS_LIST = ["car", "van", "truck", "bus"]


def create_image_annotation(
    file_path: Path, width: int, height: int, image_id: int
) -> dict:
    file_path = file_path.name
    image_annotation = {
        "file_name": file_path,
        "height": height,
        "width": width,
        "id": image_id,
    }
    return image_annotation


def eval_coco_metrics(
    anno_json: str, pred_json: str, txt_save_path: str
) -> np.ndarray:
    """
    Compare and eval pred json producing coco metrics
    """

    anno = COCO(anno_json)  # init annotations api
    pred = anno.loadRes(pred_json)  # init predictions api
    evaluator = COCOeval(anno, pred, 'bbox')

    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    # capture evaluator stats and save to file
    std_out = io.StringIO()
    with redirect_stdout(std_out):
        evaluator.summarize()
    eval_stats = std_out.getvalue()
    with open(txt_save_path, 'w', encoding="utf-8") as fwriter:
        fwriter.write(eval_stats)
    return evaluator.stats


class PatchTester:
    """
    Module for testing patches on dataset against object detection models
    """

    def __init__(self, cfg: edict) -> None:
        self.cfg = cfg
        self.dev = select_device(cfg.device)

        model = DetectMultiBackend(cfg.weights_file, device=self.dev, dnn=False, data=None, fp16=False)
        self.model = model.eval().to(self.dev)
        self.patch_transformer = PatchTransformer(
            cfg.target_size_frac, self.dev).to(self.dev)
        self.patch_applier = PatchApplier(cfg.patch_alpha).to(self.dev)

    def test(self,
             conf_thresh: float = 0.4,
             nms_thresh: float = 0.4,
             save_txt: bool = False,
             save_image: bool = False,
             save_orig_padded_image: bool = True,
             class_agnostic: bool = False,
             cls_id: Optional[int] = None,
             max_images: int = 100000) -> None:
        """
        Initiate test for properly, randomly and no-patched images
        Args:
            conf_thresh: confidence thres for successful detection/positives
            nms_thresh: nms thres
            save_txt: save the txt yolo format detections for the clean, properly and randomly patched images
            save_image: save properly and randomly patched images
            save_orig_padded_image: save orig padded images
            class_agnostic: all classes are teated the same. Use when only evaluating for obj det & not classification
            cls_id: filtering for a specific class for evaluation only
            max_images: max number of images to evaluate from inside imgdir
        """
        t0 = time.time()

        patch_size = self.cfg.patch_size
        model_in_sz = self.cfg.model_in_sz
        m_h, m_w = model_in_sz

        patch_img = Image.open(self.cfg.patchfile).convert('RGB')
        patch_img = transforms.Resize(patch_size)(patch_img)
        adv_patch_cpu = transforms.ToTensor()(patch_img)
        adv_patch = adv_patch_cpu.to(self.dev)

        clean_gt_results = []
        clean_results = []
        noise_results = []
        patch_results = []

        # make dirs
        clean_img_dir = osp.join(self.cfg.savedir, 'clean/', 'images/')
        clean_txt_dir = osp.join(self.cfg.savedir, 'clean/', 'labels/')

        proper_img_dir = osp.join(
            self.cfg.savedir, 'proper_patched/', 'images/')
        proper_txt_dir = osp.join(
            self.cfg.savedir, 'proper_patched/', 'labels/')
        random_img_dir = osp.join(
            self.cfg.savedir, 'random_patched/', 'images/')
        random_txt_dir = osp.join(
            self.cfg.savedir, 'random_patched/', 'labels/')
        jsondir = osp.join(self.cfg.savedir, 'jsons')
        print(f"Saving all outputs to {self.cfg.savedir}")
        dirs_to_create = [jsondir]
        if save_txt:
            dirs_to_create.extend([clean_txt_dir, proper_txt_dir, random_txt_dir])
        if save_image:
            dirs_to_create.extend([proper_img_dir, random_img_dir])
        if save_image and save_orig_padded_image:
            dirs_to_create.append(clean_img_dir)
        for directory in dirs_to_create:
            os.makedirs(directory, exist_ok=True)

        # dump cfg json file
        with open(os.path.join(jsondir, "cfg.json"), 'w', encoding='utf-8') as f_json:
            json.dump(self.cfg, f_json, ensure_ascii=False, indent=4)

        img_paths = glob.glob(osp.join(self.cfg.imgdir, "*"))
        img_paths = sorted(
            [p for p in img_paths if osp.splitext(p)[-1] in IMG_EXTNS])

        print("Total num images:", len(img_paths))
        img_paths = img_paths[:max_images]
        print("Considered num images:", len(img_paths))

        clean_image_annotations = []
        # to calc confusion matrixes later
        all_labels = []
        all_patch_preds = []
        all_noise_preds = []

        #######################################
        # main loop over images
        box_id = 0
        for imgfile in tqdm.tqdm(img_paths):
            img_name = osp.splitext(imgfile)[0].split('/')[-1]
            imgfile_path = Path(imgfile)
            image_id = int(imgfile_path.stem) if imgfile_path.stem.isnumeric(
            ) else imgfile_path.stem

            clean_image_annotation = create_image_annotation(
                imgfile_path, width=m_w, height=m_h, image_id=image_id)
            clean_image_annotations.append(clean_image_annotation)

            txtname = img_name + '.txt'
            txtpath = osp.join(clean_txt_dir, txtname)
            # open image and adjust to yolo input size
            img = Image.open(imgfile).convert('RGB')
            w, h = img.size

            if w == h:
                padded_img = img
            else:
                dim_to_pad = 1 if w < h else 2
                if dim_to_pad == 1:
                    padding = (h - w) / 2
                    padded_img = Image.new(
                        'RGB', (h, h), color=(127, 127, 127))
                    padded_img.paste(img, (int(padding), 0))
                else:
                    padding = (w - h) / 2
                    padded_img = Image.new(
                        'RGB', (w, w), color=(127, 127, 127))
                    padded_img.paste(img, (0, int(padding)))

            padded_img = transforms.Resize(model_in_sz)(padded_img)
            cleanname = img_name + ".png"
            # save img
            if save_image and save_orig_padded_image:
                padded_img.save(osp.join(clean_img_dir, cleanname))

            # generate a label file for the pathed image
            padded_img_tensor = transforms.ToTensor()(padded_img).unsqueeze(0).to(self.dev)
            pred = self.model(padded_img_tensor)
            boxes = non_max_suppression(pred, conf_thresh, nms_thresh)[0]
            all_labels.append(boxes.clone())
            boxes = xyxy2xywh(boxes)

            labels = []
            if save_txt:
                textfile = open(txtpath, "w+", encoding="utf-8")
            for box in boxes:
                cls_id_box = box[-1].item()
                score = box[4].item()
                x_center, y_center, width, height = box[:4]
                x_center, y_center, width, height = x_center.item(), y_center.item(), width.item(), height.item()
                labels.append(
                    [cls_id_box, x_center / m_w, y_center / m_h, width / m_w, height / m_h])
                if save_txt:
                    textfile.write(
                        f'{cls_id_box} {x_center/m_w} {y_center/m_h} {width/m_w} {height/m_h}\n')
                clean_results.append(
                    {'image_id': image_id,
                     'bbox': [x_center - width / 2, y_center - height / 2, width, height],
                     'score': score,
                     'category_id': 0 if class_agnostic else int(cls_id_box)})
                clean_gt_results.append(
                    {'id': box_id,
                     "iscrowd": 0,
                     'image_id': image_id,
                     'bbox': [x_center - width / 2, y_center - height / 2, width, height],
                     'area': width * height,
                     'category_id': 0 if class_agnostic else int(cls_id_box),
                     "segmentation": []})
                box_id += 1
            if save_txt:
                textfile.close()

            #######################################
            # Apply patch
            # use a filler ones array for no dets
            label = np.asarray(labels) if labels else np.ones([5])
            label = torch.from_numpy(label).float()
            if label.dim() == 1:
                label = label.unsqueeze(0)

            padded_img2 = transforms.ToTensor()(padded_img).to(self.dev)
            img_fake_batch = padded_img2.unsqueeze(0)
            lab_fake_batch = label.unsqueeze(0).to(self.dev)

            # Optional, Filter label_batch array to only include desired cls_id
            use_clean_boxes = False
            if cls_id is not None:
                # transform to numpy so we can filter
                lab_squeeze = label.numpy()  # lab_fake_batch.squeeze(0).numpy()
                # filter out undesired labels
                good_idxs = []
                for i, row_tmp in enumerate(lab_squeeze):
                    # rows of [1., 1., 1., 1., 1.] are filler
                    if np.array_equal(row_tmp, [1., 1., 1., 1., 1.]):
                        continue
                    # if not the desired cls_id, skip
                    elif int(row_tmp[0]) != self.cfg.cls_id:
                        continue
                    else:
                        good_idxs.append(i)
                if len(good_idxs) > 0:
                    use_clean_boxes = False
                    lab_squeeze_filt = lab_squeeze[good_idxs]
                    lab_fake_batch = torch.from_numpy(
                        lab_squeeze_filt).float().unsqueeze(0).to(self.dev)
                else:
                    use_clean_boxes = True

            # transform patch and add it to image
            if not use_clean_boxes:
                adv_batch_t = self.patch_transformer(
                    adv_patch, lab_fake_batch, model_in_sz, do_rotate=True, rand_loc=False)
                p_img_batch = self.patch_applier(img_fake_batch, adv_batch_t)
                p_img = p_img_batch.squeeze(0)
                p_img_pil = transforms.ToPILImage('RGB')(p_img.cpu())
            else:
                p_img_pil = padded_img
            properpatchedname = img_name + ".png"
            if save_image:
                p_img_pil.save(osp.join(proper_img_dir, properpatchedname))

            # generate a label file for the image with sticker
            txtname = properpatchedname.replace('.png', '.txt')
            txtpath = osp.join(proper_txt_dir, txtname)

            padded_img_tensor = transforms.ToTensor()(p_img_pil).unsqueeze(0).to(self.dev)
            pred = self.model(padded_img_tensor)
            boxes = non_max_suppression(pred, conf_thresh, nms_thresh)[0]
            all_patch_preds.append(boxes.clone())
            boxes = xyxy2xywh(boxes)

            if save_txt:
                textfile = open(txtpath, 'w+', encoding="utf-8")
            for box in boxes:
                cls_id_box = box[-1].item()
                score = box[4].item()
                x_center, y_center, width, height = box[:4]
                x_center, y_center, width, height = x_center.item(), y_center.item(), width.item(), height.item()
                if save_txt:
                    textfile.write(
                        f'{cls_id_box} {x_center/m_w} {y_center/m_h} {width/m_w} {height/m_h}\n')
                patch_results.append(
                    {'image_id': image_id,
                     'bbox': [x_center - width / 2, y_center - height / 2, width, height],
                     'score': score,
                     'category_id': 0 if class_agnostic else int(cls_id_box)})
            if save_txt:
                textfile.close()

            # create a random patch, transform it and add it to image
            random_patch = torch.rand(adv_patch_cpu.size()).to(self.dev)
            adv_batch_t = self.patch_transformer(
                random_patch, lab_fake_batch, model_in_sz, do_rotate=True, rand_loc=False)
            p_img_batch = self.patch_applier(img_fake_batch, adv_batch_t)
            p_img = p_img_batch.squeeze(0)
            p_img_pil = transforms.ToPILImage('RGB')(p_img.cpu())
            properpatchedname = img_name + ".png"
            if save_image:
                p_img_pil.save(osp.join(random_img_dir, properpatchedname))

            # generate a label file for the image with random patch
            txtname = properpatchedname.replace('.png', '.txt')
            txtpath = osp.join(random_txt_dir, txtname)

            padded_img_tensor = transforms.ToTensor()(p_img_pil).unsqueeze(0).to(self.dev)
            pred = self.model(padded_img_tensor)
            boxes = non_max_suppression(pred, conf_thresh, nms_thresh)[0]
            all_noise_preds.append(boxes.clone())
            boxes = xyxy2xywh(boxes)

            if save_txt:
                textfile = open(txtpath, 'w+', encoding="utf-8")
            for box in boxes:
                cls_id_box = box[-1].item()
                score = box[4].item()
                x_center, y_center, width, height = box[:4]
                x_center, y_center, width, height = x_center.item(), y_center.item(), width.item(), height.item()
                if save_txt:
                    textfile.write(
                        f'{cls_id_box} {x_center/m_w} {y_center/m_h} {width/m_w} {height/m_h}\n')
                noise_results.append(
                    {'image_id': image_id,
                     'bbox': [x_center - width / 2, y_center - height / 2, width, height],
                     'score': score,
                     'category_id': 0 if class_agnostic else int(cls_id_box)})
            if save_txt:
                textfile.close()

        # Calc confusion matrices if not class_agnostic
        if not class_agnostic:
            # reorder labels to (Array[M, 5]), class, x1, y1, x2, y2
            all_labels = torch.cat(all_labels)[:, [5, 0, 1, 2, 3]]
            # patch and noise labels are of shapes (Array[N, 6]), x1, y1, x2, y2, conf, class
            all_patch_preds = torch.cat(all_patch_preds)
            all_noise_preds = torch.cat(all_noise_preds)

            patch_confusion_matrix = ConfusionMatrix(len(CLASS_LIST))
            patch_confusion_matrix.process_batch(all_patch_preds, all_labels)
            noise_confusion_matrix = ConfusionMatrix(len(CLASS_LIST))
            noise_confusion_matrix.process_batch(all_noise_preds, all_labels)

            patch_confusion_matrix.plot(
                save_dir=self.cfg.savedir, names=CLASS_LIST, save_name="confusion_matrix_patch.png")
            noise_confusion_matrix.plot(
                save_dir=self.cfg.savedir, names=CLASS_LIST, save_name="confusion_matrix_noise.png")

        # add all required fields for a reference GT clean annotation
        clean_gt_results_json = {"annotations": clean_gt_results,
                                 "categories": [],
                                 "images": clean_image_annotations}
        for index, label in enumerate(CLASS_LIST, start=0):
            categories = {"supercategory": "Defect",
                          "id": index,
                          "name": label}
            clean_gt_results_json["categories"].append(categories)

        # save all json results
        clean_gt_json = osp.join(jsondir, 'clean_gt_results.json')
        clean_json = osp.join(jsondir, 'clean_results.json')
        noise_json = osp.join(jsondir, 'noise_results.json')
        patch_json = osp.join(jsondir, 'patch_results.json')

        with open(clean_gt_json, 'w', encoding="utf-8") as f_json:
            json.dump(clean_gt_results_json, f_json, ensure_ascii=False, indent=4)
        with open(clean_json, 'w', encoding="utf-8") as f_json:
            json.dump(clean_results, f_json, ensure_ascii=False, indent=4)
        with open(noise_json, 'w', encoding="utf-8") as f_json:
            json.dump(noise_results, f_json, ensure_ascii=False, indent=4)
        with open(patch_json, 'w', encoding="utf-8") as f_json:
            json.dump(patch_results, f_json, ensure_ascii=False, indent=4)

        print("####### Metrics for Images with correct patches #######")
        eval_coco_metrics(clean_gt_json, patch_json, osp.join(
            self.cfg.savedir, 'patch_map_stats.txt'))
        print("####### Metrics for Images with random noise patches #######")
        eval_coco_metrics(clean_gt_json, noise_json, osp.join(
            self.cfg.savedir, 'noise_map_stats.txt'))
        print("####### Metrics for Images with no patches for baseline tesing. Should be close to 1s #######")
        eval_coco_metrics(clean_gt_json, clean_json, osp.join(
            self.cfg.savedir, 'clean_map_stats.txt'))

        tf = time.time()
        print(f" Time to compute proper patched results = {tf - t0} seconds")


def main():
    parser = get_argparser()
    parser.add_argument('-p', '--patchfile', type=str,
                        dest="patchfile", default=None, required=True,
                        help='Path to patch image file for testing (default: %(default)s)')
    parser.add_argument('--id', '--imgdir', type=str,
                        dest="imgdir", default=None, required=True,
                        help='Path to img dir for testing (default: %(default)s)')
    parser.add_argument('--sd', '--savedir', type=str,
                        dest="savedir", default='runs/test_adversarial',
                        help='Path to save dir for saving testing results (default: %(default)s)')
    parser.add_argument('--save-txt',
                        dest="savetxt", action='store_true',
                        help='Save txt files with predicted labels in yolo fmt for later inspection')
    parser.add_argument('--save-img',
                        dest="saveimg", action='store_true',
                        help='Save images with patches for later inspection')
    parser.add_argument('--class-agnostic',
                        dest="class_agnostic", action='store_true',
                        help='All classes are teated the same. Use when only evaluating for obj det & not classification')

    args = parser.parse_args()
    cfg = load_config_object(args.config)
    cfg.patchfile = args.patchfile
    cfg.imgdir = args.imgdir
    savename = cfg.patch_name + ('_agnostic' if args.class_agnostic else '') + f'_{time.strftime("%Y%m%d-%H%M%S")}'
    cfg.savedir = osp.join(args.savedir, savename)

    tester = PatchTester(cfg)
    tester.test(save_txt=args.savetxt, save_image=args.saveimg, class_agnostic=args.class_agnostic)


if __name__ == '__main__':
    main()
