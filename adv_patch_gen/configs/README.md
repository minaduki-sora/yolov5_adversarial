# Adversarial Patch Training Config Reference


        "image_dir": "data/train/images",
        "label_dir": "data/train/labels",
        "val_image_dir": "data/val/images",     # epoch freq for running validation run. 1 means validate after every epoch. 0 or null means no val
        "use_even_odd_images": "all",           # (str), ('all', 'even', 'odd'): use images with even/odd numbers in the last char of their filenames
        "log_dir": "runs/train_adversarial",
        "tensorboard_port": 8994,
        "tensorboard_batch_log_interval": 15,
        "weights_file": "runs/weights/best.pt",
        "triplet_printfile": "triplets.csv",
        "device": "cuda:0",                     # (str): 'cpu' or 'cuda' or 'cuda:0,1,2,3'
        "use_amp": true,
        "patch_name": "base",
        "val_epoch_freq": 100,
        "patch_save_epoch_freq": 1,             # int freq for saving patches. 1 means save after every epoch
        "model_in_sz": [640, 640],              # (int, int): model input height, width
        "patch_src": "gray",                    # str: gray random, or path_to_init_patch
        "patch_size": [64, 64],                 # (int, int): must be (height, width)
        "objective_class_id": null,             # int: class id to target for adv attack. Use null for general attack for all classes
        "target_size_frac": 0.3,                # float: patch proportion size compared to bbox size. Range also accepted i.e. [0.25, 0.4]
        "use_mul_add_gau": true,                # bool: use mul & add gaussian noise or not to patches
        "mul_gau_mean": 0.5,                    # float: mul gaussian noise mean (reduces contrast) mean. Range also accepted i.e. [0.25, 0.4]
        "mul_gau_std": 0.1,                     # float: mul gaussian noise std (Adds rand noise)
        "min_pixel_area": null,                 # int: min pixel area to use for training. Pixel area chosen after resizing to model in size
        "rotate_patches": true,                 # bool: rotate patches or not
        "transform_patches": true,              # bool: add bightness, contrast and noise transforms to patches or not
        "patch_alpha": 1,                       # float: patch opacity, recommended to set to 1
        "class_list": ["class1", "class2"],
        "n_classes": 2,
        "n_epochs": 300,
        "max_labels": 48,
        "start_lr": 0.03,
        "min_tv_loss": 0.1,
        "sal_mult": 1.0,
        "tv_mult": 2.5,
        "nps_mult": 0.01,                       # float: Use 0.01 when not using sal. With sal use 0.001
        "batch_size": 8,
        "debug_mode": false,                    # bool: if yes, images with adv drawn saved during each batch
        "loss_target": "obj * cls"              # str: 'obj', 'cls', 'obj * cls'
