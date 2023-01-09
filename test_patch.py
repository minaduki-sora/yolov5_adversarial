"""
Testing code for evaluating Adversarial patches against object detection

python test_patch.py --cfg config_json_file
"""
import os
import os.path as osp
import time
import json
import glob

import numpy as np
from PIL import Image
from easydict import EasyDict as edict
import torch
from torchvision import transforms

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, xyxy2xywh
from utils.torch_utils import select_device

from adv_patch_gen.utils.config_parser import get_argparser, load_config_object
from adv_patch_gen.utils.patch import PatchApplier, PatchTransformer


IMG_EXTNS = {".png", ".jpg", ".jpeg"}


class PatchTester:
    """
    Module for testing patches on dataset against object detection models
    """

    def __init__(self, cfg: edict) -> None:
        self.cfg = cfg
        self.dev = cfg.device

        model = DetectMultiBackend(cfg.weights_file, device=select_device(
            self.dev), dnn=False, data=None, fp16=False)
        self.model = model.eval().to(self.dev)
        self.patch_transformer = PatchTransformer(cfg.target_size_frac, self.dev).to(self.dev)
        self.patch_applier = PatchApplier(cfg.patch_alpha).to(self.dev)

    def test(self,
             conf_thresh=0.4,
             nms_thresh=0.4,
             save_images=True,
             save_padded_image=True,
             cls_id=None,
             max_images=100000,
             verbose=True) -> None:
        """
        Run the test function
        """
        t0 = time.time()

        model_in_sz = self.cfg.model_in_sz[::-1]  # (w, h) to (h, w)
        mh, mw = model_in_sz
        patch_size = self.cfg.patch_size[::-1]  # (w, h) to (h, w)

        patch_img = Image.open(self.cfg.patchfile).convert('RGB')
        patch_img = transforms.Resize(patch_size)(patch_img)
        adv_patch_cpu = transforms.ToTensor()(patch_img)
        adv_patch = adv_patch_cpu.to(self.dev)

        clean_results = []
        noise_results = []
        patch_results = []

        # make dirs
        cleandir = osp.join(self.cfg.savedir, 'clean/')
        txtdir = osp.join(self.cfg.savedir, 'clean/', 'yolo-labels/')
        properdir = osp.join(self.cfg.savedir, 'proper_patched/')
        txtdir2 = osp.join(self.cfg.savedir, 'proper_patched/', 'yolo-labels/')
        randomdir = osp.join(self.cfg.savedir, 'random_patched/')
        txtdir3 = osp.join(self.cfg.savedir, 'random_patched/', 'yolo-labels/')
        jsondir = osp.join(self.cfg.savedir, 'jsons')
        for directory in (cleandir, txtdir, properdir, txtdir2, randomdir, txtdir3, jsondir):
            print(f"Creating output dir: {directory}")
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

        #######################################
        # Loop over clean images
        for i, imgfile in enumerate(img_paths):
            ti0 = time.time()
            img_name = osp.splitext(imgfile)[0].split('/')[-1]
            if (i % 50) == 0:
                print(i, "/", len(img_paths), imgfile)
            if verbose:
                print(i, "/", len(img_paths), imgfile)
            txtname = img_name + '.txt'
            txtpath = osp.join(txtdir, txtname)
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
            if save_images and save_padded_image:
                padded_img.save(osp.join(cleandir, cleanname))

            # generate a label file for the pathed image
            padded_img_tensor = transforms.ToTensor()(
                padded_img).unsqueeze(0).to(self.cfg.device)
            pred = self.model(padded_img_tensor)
            boxes = non_max_suppression(pred, conf_thresh, nms_thresh)[0]
            boxes = xyxy2xywh(boxes)

            with open(txtpath, "w+", encoding="utf-8") as textfile:
                for box in boxes:
                    cls_id_box = box[-1]
                    x_center, y_center, width, height = box[:4]
                    x_center, y_center, width, height = x_center/mw, y_center/mh, width/mw, height/mh
                    textfile.write(
                        f'{cls_id_box} {x_center} {y_center} {width} {height}\n')
                    clean_results.append({'image_id': imgfile,
                                          'bbox': [x_center.item() - width.item() / 2,
                                                   y_center.item() - height.item() / 2,
                                                   width.item(),
                                                   height.item()],
                                          'score': box[4].item(),
                                          'category_id': cls_id_box.item()})
            ti1 = time.time()
            if verbose:
                print(f" Time to compute clean results = {ti1 - ti0} seconds")

            #######################################
            # Apply patch
            # read this label file back in as a tensor
            # check to see if label file contains data.
            if osp.getsize(txtpath):
                label = np.loadtxt(txtpath)
            else:
                label = np.ones([5])
            label = torch.from_numpy(label).float()
            if label.dim() == 1:
                label = label.unsqueeze(0)

            padded_img2 = transforms.ToTensor()(padded_img).to(self.dev)
            img_fake_batch = padded_img2.unsqueeze(0)
            lab_fake_batch = label.unsqueeze(0).to(self.dev)
            if verbose:
                print(" img_fake_batch.shape", img_fake_batch.shape)
                print(" lab_fake_batch.shape", lab_fake_batch.shape)

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
            properpatchedname = img_name + "_p.png"
            if save_images:
                p_img_pil.save(osp.join(properdir, properpatchedname))

            # generate a label file for the image with sticker
            txtname = properpatchedname.replace('.png', '.txt')
            txtpath = osp.join(txtdir2, txtname)

            padded_img_tensor = transforms.ToTensor()(
                p_img_pil).unsqueeze(0).to(self.cfg.device)
            pred = self.model(padded_img_tensor)
            boxes = non_max_suppression(pred, conf_thresh, nms_thresh)[0]
            boxes = xyxy2xywh(boxes)

            with open(txtpath, 'w+', encoding="utf-8") as textfile:
                for box in boxes:
                    cls_id_box = box[-1]
                    x_center, y_center, width, height = box[:4]
                    x_center, y_center, width, height = x_center/mw, y_center/mh, width/mw, height/mh
                    textfile.write(
                        f'{cls_id_box} {x_center} {y_center} {width} {height}\n')
                    patch_results.append({'image_id': imgfile, 'bbox': [x_center.item() - width.item() / 2, y_center.item(
                    ) - height.item() / 2, width.item(), height.item()], 'score': box[4].item(), 'category_id': cls_id_box.item()})

            ti2 = time.time()
            if verbose:
                print(
                    f" Time to compute proper patched results = {ti2 - ti1} seconds")

            # create a random patch, transform it and add it to image
            random_patch = torch.rand(adv_patch_cpu.size()).to(self.dev)
            adv_batch_t = self.patch_transformer(
                random_patch, lab_fake_batch, model_in_sz, do_rotate=True, rand_loc=False)
            p_img_batch = self.patch_applier(img_fake_batch, adv_batch_t)
            p_img = p_img_batch.squeeze(0)
            p_img_pil = transforms.ToPILImage('RGB')(p_img.cpu())
            properpatchedname = img_name + "_rdp.png"
            if save_images:
                p_img_pil.save(osp.join(randomdir, properpatchedname))

            # generate a label file for the image with random patch
            txtname = properpatchedname.replace('.png', '.txt')
            txtpath = osp.join(txtdir3, txtname)

            padded_img_tensor = transforms.ToTensor()(
                p_img_pil).unsqueeze(0).to(self.cfg.device)
            pred = self.model(padded_img_tensor)
            boxes = non_max_suppression(pred, conf_thresh, nms_thresh)[0]
            boxes = xyxy2xywh(boxes)

            with open(txtpath, 'w+', encoding="utf-8") as textfile:
                for box in boxes:
                    cls_id_box = box[-1]
                    x_center, y_center, width, height = box[:4]
                    x_center, y_center, width, height = x_center/mw, y_center/mh, width/mw, height/mh
                    textfile.write(
                        f'{cls_id_box} {x_center} {y_center} {width} {height}\n')
                    noise_results.append({'image_id': imgfile, 'bbox': [x_center.item() - width.item() / 2, y_center.item(
                    ) - height.item() / 2, width.item(), height.item()], 'score': box[4].item(), 'category_id': cls_id_box.item()})

            ti3 = time.time()
            if verbose:
                print(f" Time to compute random results = {ti3 - ti2} seconds")
                print(f" Total time to compute results = {ti3 - ti0} seconds")

        # save results
        with open(osp.join(jsondir, 'clean_results.json'), 'w', encoding="utf-8") as f_json:
            json.dump(clean_results, f_json, ensure_ascii=False, indent=4)
        with open(osp.join(jsondir, 'noise_results.json'), 'w', encoding="utf-8") as f_json:
            json.dump(noise_results, f_json, ensure_ascii=False, indent=4)
        with open(osp.join(jsondir, 'patch_results.json'), 'w', encoding="utf-8") as f_json:
            json.dump(patch_results, f_json, ensure_ascii=False, indent=4)

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
                        dest="savedir", default=f'test/{time.strftime("%Y%m%d-%H%M%S")}', required=True,
                        help='Path to save dir for saving testing results (default: %(default)s)')

    args = parser.parse_args()
    cfg = load_config_object(args.config)
    cfg.patchfile = args.patchfile
    cfg.imgdir = args.imgdir
    cfg.savedir = args.savedir

    tester = PatchTester(cfg)
    tester.test()


if __name__ == '__main__':
    main()
