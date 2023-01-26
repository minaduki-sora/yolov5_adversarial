"""
Testing code for evaluating Adversarial patches against object detection
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
from adv_patch_gen.utils.common import BColors


IMG_EXTNS = {".png", ".jpg", ".jpeg"}
CLASS_LIST = ["car", "van", "truck", "bus"]


def eval_coco_metrics(anno_json: str, pred_json: str, txt_save_path: str) -> np.ndarray:
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

    @staticmethod
    def calc_asr(boxes, boxes_pred, lo_area: float = 20**2, hi_area: float = 67**2, cls_id: Optional[int] = None, class_agnostic: bool = False) -> float:
        """
        Calculate attack success rate (How many bounding boxes were hidden from the detector)
        for all predictions and for different bbox areas.
        Note cls_id is None, misclassifications are ignored and only missing detections are considered attack success.
        Args:
            boxes: torch.Tensor, first pass boxes (gt unpatched boxes) [class, x1, y1, x2, y2]
            boxes_pred: torch.Tensor, second pass boxes (patched boxes) [x1, y1, x2, y2, conf, class]
            lo_area: small bbox area threshold
            hi_area: large bbox area threshold
            cls_id: filter for a particular class
            class_agnostic: All classes are considered the same
        Return:
            attack success rates bbox area tuple: small, medium, large, all
                float, float, float, float
        """
        # if cls_id is provided and evaluation is not class agnostic then mis-clsfs count as attack success
        if cls_id is not None:
            boxes = boxes[boxes[:, 0] == cls_id]
            boxes_pred = boxes_pred[boxes_pred[:, 5] == cls_id]

        boxes_area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 4] - boxes[:, 2])
        boxes_pred_area = (boxes_pred[:, 2] - boxes_pred[:, 0]) * (boxes_pred[:, 3] - boxes_pred[:, 1])

        b_small = boxes[boxes_area < lo_area]
        bp_small = boxes_pred[boxes_pred_area < lo_area]
        b_med = boxes[torch.logical_and(boxes_area <= hi_area, boxes_area >= lo_area)]
        bp_med = boxes_pred[torch.logical_and(boxes_pred_area <= hi_area, boxes_pred_area >= lo_area)]
        b_large = boxes[boxes_area > hi_area]
        bp_large = boxes_pred[boxes_pred_area > hi_area]
        assert (bp_small.shape[0] + bp_med.shape[0] + bp_large.shape[0]) == boxes_pred.shape[0]
        assert (b_small.shape[0] + b_med.shape[0] + b_large.shape[0]) == boxes.shape[0]

        # class agnostic mode (Mis-clsfs are ignored, only non-dets matter)
        if class_agnostic:
            tp_small = bp_small.shape[0]
            tp_med = bp_med.shape[0]
            tp_large = bp_large.shape[0]
            tp_all = boxes_pred.shape[0]
        # filtering by cls_id or non class_agnostic mode (Mis-clsfs are successes)
        else:
            conf_matrix = ConfusionMatrix(len(CLASS_LIST))
            conf_matrix.process_batch(bp_small, b_small)
            tps_small, _ = conf_matrix.tp_fp()
            conf_matrix = ConfusionMatrix(len(CLASS_LIST))
            conf_matrix.process_batch(bp_med, b_med)
            tps_med, _ = conf_matrix.tp_fp()
            conf_matrix = ConfusionMatrix(len(CLASS_LIST))
            conf_matrix.process_batch(bp_large, b_large)
            tps_large, _ = conf_matrix.tp_fp()
            conf_matrix = ConfusionMatrix(len(CLASS_LIST))
            conf_matrix.process_batch(boxes_pred, boxes)
            tps_all, _ = conf_matrix.tp_fp()

            if cls_id is not None:  # consider single class, mis-clsfs or non-dets
                tp_small = tps_small[cls_id]
                tp_med = tps_med[cls_id]
                tp_large = tps_large[cls_id]
                tp_all = tps_all[cls_id]
            else:                   # non class_agnostic, mis-clsfs or non-dets
                tp_small = tps_small.sum()
                tp_med = tps_med.sum()
                tp_large = tps_large.sum()
                tp_all = tps_all.sum()

        asr_small = 1. - tp_small / (b_small.shape[0] + 1e-6)
        asr_medium = 1. - tp_med / (b_med.shape[0] + 1e-6)
        asr_large = 1. - tp_large / (b_large.shape[0] + 1e-6)
        asr_all = 1. - tp_all / (boxes.shape[0] + 1e-6)

        return max(asr_small, 0.), max(asr_medium, 0.), max(asr_large, 0.), max(asr_all, 0.)

    def _create_coco_image_annot(self, file_path: Path, width: int, height: int, image_id: int) -> dict:
        file_path = file_path.name
        image_annotation = {
            "file_name": file_path,
            "height": height,
            "width": width,
            "id": image_id,
        }
        return image_annotation

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
        proper_img_dir = osp.join(self.cfg.savedir, 'proper_patched/', 'images/')
        proper_txt_dir = osp.join(self.cfg.savedir, 'proper_patched/', 'labels/')
        random_img_dir = osp.join(self.cfg.savedir, 'random_patched/', 'images/')
        random_txt_dir = osp.join(self.cfg.savedir, 'random_patched/', 'labels/')
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
        # to calc confusion matrixes and attack success rates later
        all_labels = []
        all_patch_preds = []
        all_noise_preds = []

        #### iterate through all images ####
        box_id = 0
        for imgfile in tqdm.tqdm(img_paths):
            img_name = osp.splitext(imgfile)[0].split('/')[-1]
            imgfile_path = Path(imgfile)
            image_id = int(imgfile_path.stem) if imgfile_path.stem.isnumeric(
            ) else imgfile_path.stem

            clean_image_annotation = self._create_coco_image_annot(
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

            #######################################
            # generate labels for the patched image
            padded_img_tensor = transforms.ToTensor()(padded_img).unsqueeze(0).to(self.dev)
            pred = self.model(padded_img_tensor)
            boxes = non_max_suppression(pred, conf_thresh, nms_thresh)[0]
            # if doing targeted class performance check, ignore non target classes
            if cls_id is not None:
                boxes = boxes[boxes[:, -1] == cls_id]
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

            # use a filler ones array for no dets
            label = np.asarray(labels) if labels else np.ones([5])
            label = torch.from_numpy(label).float()
            if label.dim() == 1:
                label = label.unsqueeze(0)

            #######################################
            # Apply proper patches
            padded_img_copy = transforms.ToTensor()(padded_img).to(self.dev)
            img_fake_batch = padded_img_copy.unsqueeze(0)
            lab_fake_batch = label.unsqueeze(0).to(self.dev)

            if len(lab_fake_batch) == 1 and np.array_equal(lab_fake_batch[0], [1., 1., 1., 1., 1.]):
                # no det, use images without patches
                p_img_pil = padded_img
            else:
                # transform patch and add it to image
                adv_batch_t = self.patch_transformer(
                    adv_patch, lab_fake_batch, model_in_sz, do_rotate=True, rand_loc=False)
                p_img_batch = self.patch_applier(img_fake_batch, adv_batch_t)
                p_img = p_img_batch.squeeze(0)
                p_img_pil = transforms.ToPILImage('RGB')(p_img.cpu())

            properpatchedname = img_name + ".png"
            if save_image:
                p_img_pil.save(osp.join(proper_img_dir, properpatchedname))

            # generate a label file for the image with sticker
            txtname = properpatchedname.replace('.png', '.txt')
            txtpath = osp.join(proper_txt_dir, txtname)

            padded_img_tensor = transforms.ToTensor()(p_img_pil).unsqueeze(0).to(self.dev)
            pred = self.model(padded_img_tensor)
            boxes = non_max_suppression(pred, conf_thresh, nms_thresh)[0]
            # if doing targeted class performance check, ignore non target classes
            if cls_id is not None:
                boxes = boxes[boxes[:, -1] == cls_id]
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

            #######################################
            # Apply random patches
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
            # if doing targeted class performance check, ignore non target classes
            if cls_id is not None:
                boxes = boxes[boxes[:, -1] == cls_id]
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

        # reorder labels to (Array[M, 5]), class, x1, y1, x2, y2
        all_labels = torch.cat(all_labels)[:, [5, 0, 1, 2, 3]]
        # patch and noise labels are of shapes (Array[N, 6]), x1, y1, x2, y2, conf, class
        all_patch_preds = torch.cat(all_patch_preds)
        all_noise_preds = torch.cat(all_noise_preds)

        # Calc confusion matrices if not class_agnostic
        if not class_agnostic:
            patch_confusion_matrix = ConfusionMatrix(len(CLASS_LIST))
            patch_confusion_matrix.process_batch(all_patch_preds, all_labels)
            noise_confusion_matrix = ConfusionMatrix(len(CLASS_LIST))
            noise_confusion_matrix.process_batch(all_noise_preds, all_labels)

            patch_confusion_matrix.plot(save_dir=self.cfg.savedir, names=CLASS_LIST, save_name="conf_matrix_patch.png")
            noise_confusion_matrix.plot(save_dir=self.cfg.savedir, names=CLASS_LIST, save_name="conf_matrix_noise.png")

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

        patch_txt_path = osp.join(self.cfg.savedir, 'patch_map_stats.txt')
        noise_txt_path = osp.join(self.cfg.savedir, 'noise_map_stats.txt')
        clean_txt_path = osp.join(self.cfg.savedir, 'clean_map_stats.txt')

        print(f"{BColors.HEADER}### Metrics for images with no patches for baseline. Should be ~1 ###{BColors.ENDC}")
        eval_coco_metrics(clean_gt_json, clean_json, clean_txt_path)

        print(f"{BColors.HEADER}### Metrics for images with correct patches ###{BColors.ENDC}")
        eval_coco_metrics(clean_gt_json, patch_json, patch_txt_path)

        asr_s, asr_m, asr_l, asr_a = PatchTester.calc_asr(all_labels, all_patch_preds, cls_id=cls_id, class_agnostic=class_agnostic)
        with open(patch_txt_path, 'a', encoding="utf-8") as f_patch:
            asr_str = ''
            asr_str += f" Attack success rate (@conf={conf_thresh}) | class_agnostic={class_agnostic} | area= small | = {asr_s:.3f}\n"
            asr_str += f" Attack success rate (@conf={conf_thresh}) | class_agnostic={class_agnostic} | area=medium | = {asr_m:.3f}\n"
            asr_str += f" Attack success rate (@conf={conf_thresh}) | class_agnostic={class_agnostic} | area= large | = {asr_l:.3f}\n"
            asr_str += f" Attack success rate (@conf={conf_thresh}) | class_agnostic={class_agnostic} | area=   all | = {asr_a:.3f}\n"
            print(asr_str)
            f_patch.write(asr_str)

        print(f"{BColors.HEADER}### Metrics for images with random noise patches ###{BColors.ENDC}")
        eval_coco_metrics(clean_gt_json, noise_json, noise_txt_path)

        asr_s, asr_m, asr_l, asr_a = PatchTester.calc_asr(all_labels, all_noise_preds, cls_id=cls_id, class_agnostic=class_agnostic)
        with open(noise_txt_path, 'a', encoding="utf-8") as f_noise:
            asr_str = ''
            asr_str += f" Attack success rate (@conf={conf_thresh}) | class_agnostic={class_agnostic} | area= small | = {asr_s:.3f}\n"
            asr_str += f" Attack success rate (@conf={conf_thresh}) | class_agnostic={class_agnostic} | area=medium | = {asr_m:.3f}\n"
            asr_str += f" Attack success rate (@conf={conf_thresh}) | class_agnostic={class_agnostic} | area= large | = {asr_l:.3f}\n"
            asr_str += f" Attack success rate (@conf={conf_thresh}) | class_agnostic={class_agnostic} | area=   all | = {asr_a:.3f}\n"
            print(asr_str)
            f_noise.write(asr_str)

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
    parser.add_argument('--target-class', type=int,
                        dest="target_class", default=None, required=False,
                        help='Target specific class with id for misclassification test (default: %(default)s)')

    args = parser.parse_args()
    cfg = load_config_object(args.config)
    cfg.patchfile = args.patchfile
    cfg.imgdir = args.imgdir

    savename = cfg.patch_name + ('_agnostic' if args.class_agnostic else '')
    if args.class_agnostic and args.target_class is not None:
        print(f"""{BColors.WARNING}WARNING:{BColors.ENDC} target_class and class_agnostic are both set.
              Target_class will be ignored and metrics will be class agnostic. Only set either.""")
        args.target_class = None
    else:
        savename += (f'_tc{args.target_class}' if args.target_class is not None else '')
    savename += f'_{time.strftime("%Y%m%d-%H%M%S")}'
    cfg.savedir = osp.join(args.savedir, savename)

    print(f"{BColors.OKBLUE} Test Arguments: {args} {BColors.ENDC}")
    tester = PatchTester(cfg)
    tester.test(save_txt=args.savetxt, save_image=args.saveimg, class_agnostic=args.class_agnostic, cls_id=args.target_class)


if __name__ == '__main__':
    main()
