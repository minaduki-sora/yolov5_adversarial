"""
Training code for Adversarial patch training

To run:
python train_patch.py [model]
e.g.:    python train_patch.py visdrone_v1_4cat_obj_only_tiny_gray_v1

"""
import os
import time
import json
import logging
import subprocess
from typing import Optional

from PIL import Image
from tqdm import tqdm
from easydict import EasyDict as edict

import torch
import torch.nn.functional as F
from torch import optim, autograd
from torchvision import transforms

from tensorboardX import SummaryWriter

from models.common import DetectMultiBackend
from utils.torch_utils import select_device

from adv_patch_gen.utils.config_parser import get_config_from_args
from adv_patch_gen.utils.dataset import YOLODataset
from adv_patch_gen.utils.patch import PatchApplier, PatchTransformer
from adv_patch_gen.utils.loss import MaxProbExtractor, SaliencyLoss, TotalVariationLoss, NPSLoss


class PatchTrainer:
    """
    Module for training on dataset to generate adv patches
    """

    def __init__(self, config: edict):
        self.config = config
        self.dev = config.device

        model = DetectMultiBackend(config.weights_file, device=select_device(self.dev), dnn=False, data=None, fp16=False)
        self.model = model.eval().to(self.dev)
        self.patch_transformer = PatchTransformer(config.target_size_frac).to(self.dev)
        self.patch_applier = PatchApplier(config.patch_alpha).to(self.dev)
        self.prob_extractor = MaxProbExtractor(config).to(self.dev)
        self.sal_loss = SaliencyLoss().to(self.dev)
        self.nps_loss = NPSLoss(config.triplet_printfile, config.patch_size).to(self.dev)
        self.tv_loss = TotalVariationLoss().to(self.dev)
        self.writer = self.init_tensorboard(config.patch_name)

        # load dataset
        self.train_loader = torch.utils.data.DataLoader(
            YOLODataset(config.image_dir,
                        config.label_dir,
                        config.max_labels,
                        config.model_in_sz,
                        shuffle=True),
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=10)
        self.epoch_length = len(self.train_loader)
        print(f'One epoch is {len(self.train_loader)}')

    def init_tensorboard(self, name: Optional[str] = None, port: int = 8994):
        """
        Initialize tensorboard with optional name
        """
        # subprocess.Popen(['tensorboard', '--logdir=runs', f'--port={port}'])
        if name is not None:
            time_str = time.strftime("%Y%m%d-%H%M%S")
            return SummaryWriter(f'runs/{time_str}_{name}')
        else:
            return SummaryWriter()

    def train(self) -> None:
        """
        Optimize a patch to generate an adversarial example.
        """

        # make output dir
        patch_dir = os.path.join('./saved_patches', self.config.patch_name + f'_{time.strftime("%Y%m%d-%H%M%S")}')
        os.makedirs(patch_dir, exist_ok=True)
        log_file = os.path.join(patch_dir, self.config.patch_name + '_log.txt')
        # dump config json file
        with open(os.path.join(patch_dir, "config.json"), 'w', encoding='utf-8') as f:
            json.dump(self.config, f, ensure_ascii=False, indent=4)

        # fix loss targets
        loss_target = self.config.loss_target
        if loss_target == "obj":
            self.config.loss_target = lambda obj, cls: obj
        elif loss_target == "cls":
            self.config.loss_target = lambda obj, cls: cls
        elif loss_target == "obj * cls":
            self.config.loss_target = lambda obj, cls: obj * cls
        else:
            raise NotImplementedError(f"Loss target {loss_target} has not been implemented")

        # python logging
        ###############################################################################
        # https://docs.python.org/3/howto/logging-cookbook.html#logging-to-multiple-destinations
        # set up logging to file - see previous section for more details
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
            datefmt='%m-%d %H:%M',
            filename=log_file,
            filemode='w')
        # define a Handler which writes INFO messages or higher to the sys.stderr
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        # set a format which is simpler for console use
        formatter = logging.Formatter(
            '%(name)-12s: %(levelname)-8s %(message)s')
        # tell the handler to use this format
        console.setFormatter(formatter)
        # add the handler to the root logger
        logging.getLogger('').addHandler(console)
        ###############################################################################

        # print config values to log
        logging.info("patch_dir: %s", patch_dir)
        logging.info("config_dict:")
        config_dict = {key: value for key, value in self.config.__dict__.items(
        ) if not key.startswith('__') and not callable(key)}
        for k, v in config_dict.items():
            logging.info("key=%s\t val=%s", k, v)

        # Generate init patch
        if self.config.patch_src == 'gray':
            adv_patch_cpu = self.generate_patch("gray")
        elif self.config.patch_src == 'random':
            adv_patch_cpu = self.generate_patch("random")
        else:
            adv_patch_cpu = self.read_image(self.config.patch_src)
        adv_patch_cpu.requires_grad = True

        optimizer = optim.Adam([adv_patch_cpu], lr=self.config.start_lr, amsgrad=True)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50)

        et0 = time.time()
        for epoch in range(self.config.n_epochs):
            out_patch_path = os.path.join(
                patch_dir, self.config.patch_name + '_epoch' + str(epoch) + '.jpg')
            ep_det_loss = 0
            ep_nps_loss = 0
            ep_tv_loss = 0
            ep_loss = 0
            bt0 = time.time()
            for i_batch, (img_batch, lab_batch) in tqdm(enumerate(self.train_loader), 
                                                        desc=f'Running epoch {epoch}',
                                                        total=self.epoch_length):
                with autograd.detect_anomaly():
                    img_batch = img_batch.to(self.dev)
                    lab_batch = lab_batch.to(self.dev)
                    if (i_batch % 100) == 0:
                        logging.info('TRAINING EPOCH %i, BATCH %i', epoch, i_batch)
                    adv_patch = adv_patch_cpu.to(self.dev)
                    adv_batch_t = self.patch_transformer(
                        adv_patch, lab_batch, self.config.model_in_sz, do_rotate=True, rand_loc=False)
                    p_img_batch = self.patch_applier(
                        img_batch, adv_batch_t)
                    p_img_batch = F.interpolate(
                        p_img_batch, (self.config.model_in_sz[1], self.config.model_in_sz[0]))

                    img = p_img_batch[1, :, :, ]
                    img = transforms.ToPILImage()(img.detach().cpu())
                    # img.show()

                    output = self.model(p_img_batch)[0]
                    max_prob = self.prob_extractor(output)
                    nps = self.nps_loss(adv_patch)
                    tv = self.tv_loss(adv_patch)

                    nps_loss = nps * self.config.nps_mult
                    tv_loss = tv * self.config.tv_mult
                    det_loss = torch.mean(max_prob)
                    loss = det_loss + nps_loss + \
                        torch.max(tv_loss, torch.tensor(0.1).to(self.dev))

                    ep_det_loss += det_loss.detach().cpu().numpy()
                    ep_nps_loss += nps_loss.detach().cpu().numpy()
                    ep_tv_loss += tv_loss.detach().cpu().numpy()
                    ep_loss += loss

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    # keep patch in image range
                    adv_patch_cpu.data.clamp_(0, 1)

                    bt1 = time.time()
                    if i_batch % 5 == 0:
                        iteration = self.epoch_length * epoch + i_batch
                        self.writer.add_scalar(
                            'total_loss', loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar(
                            'loss/det_loss', det_loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar(
                            'loss/nps_loss', nps_loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar(
                            'loss/tv_loss', tv_loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar(
                            'misc/epoch', epoch, iteration)
                        self.writer.add_scalar(
                            'misc/learning_rate', optimizer.param_groups[0]["lr"], iteration)
                        self.writer.add_image(
                            'patch', adv_patch_cpu, iteration)
                    if i_batch + 1 >= len(self.train_loader):
                        print('\n')
                    else:
                        del adv_batch_t, output, max_prob, det_loss, p_img_batch, nps_loss, tv_loss, loss
                        torch.cuda.empty_cache()
                    bt0 = time.time()
            et1 = time.time()
            ep_det_loss = ep_det_loss/len(self.train_loader)
            ep_nps_loss = ep_nps_loss/len(self.train_loader)
            ep_tv_loss = ep_tv_loss/len(self.train_loader)
            ep_loss = ep_loss/len(self.train_loader)

            scheduler.step(ep_loss)
            if True:
                logging.info("  EPOCH NR: %s", epoch)
                logging.info("EPOCH LOSS: %s", ep_loss)
                logging.info("  DET LOSS: %s", ep_det_loss)
                logging.info("  NPS LOSS: %s", ep_nps_loss)
                logging.info("   TV LOSS: %s", ep_tv_loss)
                logging.info("EPOCH TIME: %s", et1 - et0)

                im = transforms.ToPILImage('RGB')(adv_patch_cpu)
                im.save(out_patch_path)
                del adv_batch_t, output, max_prob, det_loss, p_img_batch, nps_loss, tv_loss, loss
                torch.cuda.empty_cache()
            et0 = time.time()

    def generate_patch(self, patch_type: str) -> torch.Tensor:
        """
        Generate a random patch as a starting point for optimization.

        Arguments:
            patch_type: Can be 'gray' or 'random'. Whether or not generate a gray or a random patch.
        """
        p_w, p_h = self.config.patch_size
        if patch_type == 'gray':
            adv_patch_cpu = torch.full((3, p_h, p_w), 0.5)
        elif patch_type == 'random':
            adv_patch_cpu = torch.rand((3, p_h, p_w))

        return adv_patch_cpu

    def read_image(self, path):
        """
        Read an input image to be used as a patch

        :param path: Path to the image to be read.
        :return: Returns the transformed patch as a pytorch Tensor.
        """
        patch_img = Image.open(path).convert('RGB')
        transforms_resize = transforms.Resize(self.config.patch_size)
        patch_img = transforms_resize(patch_img)
        adv_patch_cpu = transforms.ToTensor()(patch_img)
        return adv_patch_cpu


def main():
    config = get_config_from_args()
    trainer = PatchTrainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
