"""
Create argparse options for config files
"""
import json
import argparse
from easydict import EasyDict as edict


def get_config_object(cfg_path: str) -> edict:
    """
    Loads a config json and returns a edict object
    """
    with open(cfg_path, 'r', encoding="utf-8") as json_file:
        cfg_dict = json.load(json_file)
    
    return edict(cfg_dict)


def get_config_from_args() -> edict:
    """
    Get edict from chosen config file
    """
    parser = argparse.ArgumentParser(
        description="Config file load for training adv patches")
    parser.add_argument('--cfg', '--config', type=str, dest="config", default="adv_patch_gen/configs/base.json",
                        help='Path to config file to use for adv patch generation (default: %(default)s)')
    args = parser.parse_args()

    return get_config_object(args.config)

