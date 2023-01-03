"""
Create argparse options for config files
"""
import argparse
import adv_patch_gen.configs as configs


CONFIG_DICT = {"base": configs.BaseConfig}


def get_config_from_args() -> dict:
    parser = argparse.ArgumentParser(
        description="Config file load for training adv patches")
    parser.add_argument('-c', '--config', type=str, dest="config", required=True,
                        choices=CONFIG_DICT.keys(),
                        help='Configurations to use for adv patch generation (default: %(default)s)')
    args = parser.parse_args()

    return CONFIG_DICT[args.config]()

