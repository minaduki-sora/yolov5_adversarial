""" Modules for creating adversarial object patch """
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from adv_patch_gen.utils.median_pool import MedianPool2d


class PatchTransformer(nn.Module):
    """PatchTransformer: transforms batch of patches

    Module providing the functionality necessary to transform a batch of patches, randomly adjusting brightness and
    contrast, adding random amount of noise, and rotating randomly. Resizes patches according to as size based on the
    batch of labels, and pads them to the dimension of an image.

    """

    def __init__(self, target_size_frac: float = 0.35, dev: str = "cuda"):

        super(PatchTransformer, self).__init__()
        self.target_size_frac = target_size_frac  # Originally 0.2
        self.dev=dev
        self.min_contrast = 0.8
        self.max_contrast = 1.2
        self.min_brightness = -0.1
        self.max_brightness = 0.1
        self.noise_factor = 0.10
        self.minangle = -20 / 180 * math.pi
        self.maxangle = 20 / 180 * math.pi
        self.medianpooler = MedianPool2d(7, same=True)

    def forward(self, adv_patch, lab_batch, img_size, do_rotate=True, rand_loc=True):
        adv_patch = self.medianpooler(adv_patch.unsqueeze(0))
        # Determine size of padding
        pad = (img_size[0] - adv_patch.size(-1)) / 2
        # Make a batch of patches
        adv_patch = adv_patch.unsqueeze(0)
        adv_batch = adv_patch.expand(
            lab_batch.size(0), lab_batch.size(1), -1, -1, -1)
        batch_size = torch.Size((lab_batch.size(0), lab_batch.size(1)))

        # Contrast, brightness and noise transforms

        # Create random contrast tensor
        contrast = torch.FloatTensor(batch_size).uniform_(
            self.min_contrast, self.max_contrast).to(self.dev)
        contrast = contrast.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        contrast = contrast.expand(-1, -1, adv_batch.size(-3),
                                   adv_batch.size(-2), adv_batch.size(-1))
        contrast = contrast.to(self.dev)

        # Create random brightness tensor
        brightness = torch.FloatTensor(batch_size).uniform_(
            self.min_brightness, self.max_brightness).to(self.dev)
        brightness = brightness.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        brightness = brightness.expand(-1, -1, adv_batch.size(-3),
                                       adv_batch.size(-2), adv_batch.size(-1))
        brightness = brightness.to(self.dev)

        # Create random noise tensor
        noise = torch.FloatTensor(
            adv_batch.size()).uniform_(-1, 1).to(self.dev) * self.noise_factor

        # Apply contrast/brightness/noise, clamp
        adv_batch = adv_batch * contrast + brightness + noise

        adv_batch = torch.clamp(adv_batch, 0.000001, 0.99999)

        # Where the label class_id is 1 we don't want a patch (padding) --> fill mask with zero's
        cls_ids = torch.narrow(lab_batch, 2, 0, 1)
        cls_mask = cls_ids.expand(-1, -1, 3)
        cls_mask = cls_mask.unsqueeze(-1)
        cls_mask = cls_mask.expand(-1, -1, -1, adv_batch.size(3))
        cls_mask = cls_mask.unsqueeze(-1)
        cls_mask = cls_mask.expand(-1, -1, -1, -1, adv_batch.size(4))
        msk_batch = torch.FloatTensor(cls_mask.size()).fill_(1).to(self.dev) - cls_mask

        # Pad patch and mask to image dimensions
        patch_pad = nn.ConstantPad2d(
            (int(pad + 0.5), int(pad), int(pad + 0.5), int(pad)), 0)
        adv_batch = patch_pad(adv_batch)
        msk_batch = patch_pad(msk_batch)

        # Rotation and rescaling transforms
        anglesize = (lab_batch.size(0) * lab_batch.size(1))
        if do_rotate:
            angle = torch.FloatTensor(anglesize).uniform_(self.minangle, self.maxangle).to(self.dev)
        else:
            angle = torch.FloatTensor(anglesize).fill_(0).to(self.dev)

        # Resizes and rotates
        current_patch_size = adv_patch.size(-1)
        lab_batch_scaled = torch.FloatTensor(lab_batch.size()).fill_(0).to(self.dev)
        lab_batch_scaled[:, :, 1] = lab_batch[:, :, 1] * img_size[0]
        lab_batch_scaled[:, :, 2] = lab_batch[:, :, 2] * img_size[0]
        lab_batch_scaled[:, :, 3] = lab_batch[:, :, 3] * img_size[0]
        lab_batch_scaled[:, :, 4] = lab_batch[:, :, 4] * img_size[0]
        target_size = torch.sqrt(((lab_batch_scaled[:, :, 3].mul(self.target_size_frac)) ** 2)
                                 + ((lab_batch_scaled[:, :, 4].mul(self.target_size_frac)) ** 2))

        target_x = lab_batch[:, :, 1].view(np.prod(batch_size))
        target_y = lab_batch[:, :, 2].view(np.prod(batch_size))
        targetoff_x = lab_batch[:, :, 3].view(np.prod(batch_size))
        targetoff_y = lab_batch[:, :, 4].view(np.prod(batch_size))
        if (rand_loc):
            off_x = targetoff_x * \
                (torch.FloatTensor(targetoff_x.size()).uniform_(-0.4, 0.4).to(self.dev))
            target_x = target_x + off_x
            off_y = targetoff_y * \
                (torch.FloatTensor(targetoff_y.size()).uniform_(-0.4, 0.4).to(self.dev))
            target_y = target_y + off_y
        scale = target_size / current_patch_size
        scale = scale.view(anglesize)

        s = adv_batch.size()
        adv_batch = adv_batch.view(s[0] * s[1], s[2], s[3], s[4])
        msk_batch = msk_batch.view(s[0] * s[1], s[2], s[3], s[4])

        tx = (-target_x+0.5)*2
        ty = (-target_y+0.5)*2
        sin = torch.sin(angle)
        cos = torch.cos(angle)

        # Theta = rotation,rescale matrix
        theta = torch.FloatTensor(anglesize, 2, 3).fill_(0).to(self.dev)
        theta[:, 0, 0] = cos/scale
        theta[:, 0, 1] = sin/scale
        theta[:, 0, 2] = tx*cos/scale+ty*sin/scale
        theta[:, 1, 0] = -sin/scale
        theta[:, 1, 1] = cos/scale
        theta[:, 1, 2] = -tx*sin/scale+ty*cos/scale

        grid = F.affine_grid(theta, adv_batch.shape)
        adv_batch_t = F.grid_sample(adv_batch, grid)
        msk_batch_t = F.grid_sample(msk_batch, grid)

        adv_batch_t = adv_batch_t.view(s[0], s[1], s[2], s[3], s[4])
        msk_batch_t = msk_batch_t.view(s[0], s[1], s[2], s[3], s[4])

        adv_batch_t = torch.clamp(adv_batch_t, 0.000001, 0.999999)
        #img = msk_batch_t[0, 0, :, :, :].detach().cpu()
        #img = transforms.ToPILImage()(img)
        # img.show()
        # exit()

        return adv_batch_t * msk_batch_t


class PatchApplier(nn.Module):
    """PatchApplier: applies adversarial patches to images.

    Module providing the functionality necessary to apply a patch to all detections in all images in the batch.
    The patch (adv_batch) has the same size as the image, just is zero everywhere there isn't a patch.
    If patch_alpha == 1 (default), just overwrite the background image values with the patch values.
    Else, blend the patch with the image 
    See: https://learnopencv.com/alpha-blending-using-opencv-cpp-python/
         https://stackoverflow.com/questions/49737541/merge-two-images-with-alpha-channel/49738078
        I = \alpha F + (1 - \alpha) B
            F = foregraound (patch, or adv_batch)
            B = background (image, or img_batch)
    """

    def __init__(self, patch_alpha: float = 1):
        super(PatchApplier, self).__init__()
        self.patch_alpha = patch_alpha

    def forward(self, img_batch, adv_batch):
        advs = torch.unbind(adv_batch, 1)
        for adv in advs:
            # replace image values with patch values
            if self.patch_alpha == 1:
                img_batch = torch.where((adv == 0), img_batch, adv)
            # alpha blend
            else:
                # get combo of image and adv
                alpha_blend = self.patch_alpha * adv + \
                    (1.0 - self.patch_alpha) * img_batch
                # apply alpha blend where the patch is non-zero
                img_batch = torch.where((adv == 0), img_batch, alpha_blend)

        return img_batch
