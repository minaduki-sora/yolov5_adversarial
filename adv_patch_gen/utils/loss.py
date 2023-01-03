"""
Loss functions used in patch generation
"""
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


class MaxProbExtractor(nn.Module):
    """MaxProbExtractor: extracts max class probability for class from YOLO output.

    Module providing the functionality necessary to extract the max class probability for one class from YOLO output.

    """

    def __init__(self, config):
        super(MaxProbExtractor, self).__init__()
        self.config = config

    def forward(self, output: torch.Tensor):
        """
        output must be of the shape [batch, -1, 5 + num_cls]
        """
        # get values neccesary for transformation
        assert (output.size(-1) == (5 + self.config.n_classes))

        output_class_scores = output[:, :, 5:5 + self.config.n_classes]  # [batch, -1, n_classes]
        # norm probs for object classes to [0, 1]
        confs_for_class = torch.nn.Softmax(dim=2)(output_class_scores)
        output_objectness = torch.sigmoid(output[:, :, 4]).unsqueeze(-1)  # [batch, -1, 1]
        confs_if_object = self.config.loss_target(output_objectness, confs_for_class)
        max_conf, _ = torch.max(confs_if_object, dim=1)
        return max_conf


class SaliencyLoss(nn.Module):
    """Implementation of the colorfulness metric as the saliency loss.
    The smaller the value, the less colorful the image.
    Reference: https://infoscience.epfl.ch/record/33994/files/HaslerS03.pdf
    """

    def __init__(self):
        super(SaliencyLoss, self).__init__()

    def forward(self, adv_patch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            adv_patch: Float Tensor of shape [C, H, W] where C=3 (R, G, B channels)
        """
        assert adv_patch.shape[0] == 3
        r, g, b = adv_patch
        rg = r - g
        yb = 0.5 * (r + g) - b

        mu_rg, sigma_rg = torch.mean(rg), torch.std(rg)
        mu_yb, sigma_yb = torch.mean(yb), torch.std(yb)
        sl = torch.sqrt(sigma_rg**2 + sigma_yb**2) + \
            (0.3 * torch.sqrt(mu_rg**2 + mu_yb**2))
        return sl / torch.numel(adv_patch)


class TotalVariationLoss(nn.Module):
    """TotalVariationLoss: calculates the total variation of a patch.
    Module providing the functionality necessary to calculate the total vatiation (TV) of an adversarial patch.
    Reference: https://en.wikipedia.org/wiki/Total_variation
    """

    def __init__(self):
        super(TotalVariationLoss, self).__init__()

    def forward(self, adv_patch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            adv_patch: Tensor of shape [C, H, W] 
        """
        # calc diff in patch rows
        tvcomp_r = torch.sum(
            torch.abs(adv_patch[:, :, 1:] - adv_patch[:, :, :-1]+0.000001), dim=0)
        tvcomp_r = torch.sum(torch.sum(tvcomp_r, dim=0), dim=0)
        # calc diff in patch columns
        tvcomp_c = torch.sum(
            torch.abs(adv_patch[:, 1:, :] - adv_patch[:, :-1, :]+0.000001), dim=0)
        tvcomp_c = torch.sum(torch.sum(tvcomp_c, dim=0), dim=0)
        tv = tvcomp_r + tvcomp_c
        return tv / torch.numel(adv_patch)


class NPSLoss(nn.Module):
    """NMSLoss: calculates the non-printability-score loss of a patch.
    Module providing the functionality necessary to calculate the non-printability score (NMS) of an adversarial patch.
    However, a summation of the differences is used instead of the total product to calc the NPSLoss
    Reference: https://users.ece.cmu.edu/~lbauer/papers/2016/ccs2016-face-recognition.pdf
        Args: 
            triplet_scores_fpath: str, path to csv file with RGB triplets sep by commas in newlines
            size: Tuple[int, int], Tuple with width, height of the patch
    """

    def __init__(self, triplet_scores_fpath: str, size: Tuple[int, int]):
        super(NPSLoss, self).__init__()
        self.printability_array = nn.Parameter(self.get_printability_array(
            triplet_scores_fpath, size), requires_grad=False)

    def forward(self, adv_patch):
        # calculate euclidian distance between colors in patch and colors in printability_array
        # square root of sum of squared difference
        color_dist = (adv_patch - self.printability_array + 0.000001)
        color_dist = color_dist ** 2
        color_dist = torch.sum(color_dist, 1) + 0.000001
        color_dist = torch.sqrt(color_dist)
        # use the min distance
        color_dist_prod = torch.min(color_dist, 0)[0]
        # calculate the nps by summing over all pixels
        nps_score = torch.sum(color_dist_prod, 0)
        nps_score = torch.sum(nps_score, 0)
        return nps_score / torch.numel(adv_patch)

    def get_printability_array(self, triplet_scores_fpath: str, size: Tuple[int, int]) -> torch.Tensor:
        """
        Get printability tensor array holding the rgb triplets (range [0,1]) loaded from triplet_scores_fpath
        Args: 
            triplet_scores_fpath: str, path to csv file with RGB triplets sep by commas in newlines
            size: Tuple[int, int], Tuple with width, height of the patch
        """
        ref_triplet_list = []
        # read in reference printability triplets into a list
        with open(triplet_scores_fpath) as f:
            for line in f:
                ref_triplet_list.append(line.strip().split(","))

        w, h = size
        printability_array = []
        for ref_triplet in ref_triplet_list:
            r, g, b = map(float, ref_triplet)
            ref_tensor_img = torch.stack([torch.full((h, w), r),
                                          torch.full((h, w), g),
                                          torch.full((h, w), b)])
            printability_array.append(ref_tensor_img.float())
        return torch.stack(printability_array)

    def get_printability_array_old(self, triplet_scores_fpath: str, patch_side_len: int) -> torch.Tensor:
        """
        Get printability tensor array holding the rgb triplets (range [0,1]) loaded from triplet_scores_fpath
        Args: 
            triplet_scores_fpath: str, path to csv file with RGB triplets sep by commas in newlines
            patch_side_len: int, length of the sides of the patch
        """
        side = patch_side_len
        ref_triplet_list = []

        # read in reference printability triplets and put them in a list
        with open(triplet_scores_fpath) as f:
            for line in f:
                ref_triplet_list.append(line.strip().split(","))

        printability_array = []
        for ref_triplet in ref_triplet_list:
            ref_imgs = []
            r, g, b = ref_triplet
            ref_imgs.append(np.full((side, side), r))
            ref_imgs.append(np.full((side, side), g))
            ref_imgs.append(np.full((side, side), b))
            printability_array.append(ref_imgs)

        printability_array = np.asarray(printability_array)
        printability_array = np.float32(printability_array)
        pa = torch.from_numpy(printability_array)
        print(pa.shape, pa.dtype)
        return pa
