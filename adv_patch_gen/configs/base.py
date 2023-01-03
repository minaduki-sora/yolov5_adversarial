from torch import optim


class BaseConfig:
    """
    Default parameters for all config files.
    """

    def __init__(self):
        self.image_dir = "YOLO_FROMAT_DATASET/images"
        self.label_dir = "YOLO_FROMAT_DATASET/labels"
        self.weights_file = "runs/train/s_95_coco_e100/weights/best.pt"
        self.triplet_printfile = "adv_patch_gen/utils/30_rgb_triplets.csv"
        self.device = "cuda:0"  # "cpu", "cuda", "cuda:dev_num"
        self.model_in_sz = (640, 640)

        self.patch_name = 'base'
        self.patch_src = "gray"  # "gray", "random", or path to init patch
        self.patch_size = (64, 64)  # (width, height)
        self.target_size_frac = 0.2
        self.patch_alpha = 1  # range [0, 1]

        self.n_classes = 4
        self.n_epochs = 1000  # orig 10000
        self.max_labels = 100  # orig 14
        self.start_lr = 0.03
        self.scheduler_factory = lambda x: optim.lr_scheduler.ReduceLROnPlateau(x, 'min', patience=50)
        self.max_tv = 0
        self.tv_mult = 2.5
        self.nps_mult = 0.01
        self.batch_size = 4
        # obj: only target objectness loss, cls: only target class-score loss, obj*cls: target both class & objectness loss
        self.loss_target = lambda obj, cls: obj * cls
