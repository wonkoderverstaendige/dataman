# generate .fet file from .fd files
#           OR
# if those features are missing, generate them with dm features!
# - generate/load features
# - run klustakwik
# - read clusters, build initial clustering report
import argparse
import logging
import shutil
from pathlib import Path

import pkg_resources
import yaml

logger = logging.getLogger(__name__)


def load_yaml(yaml_path):
    """Load content of YAML file as dict
    """
    yaml_path = Path(yaml_path).resolve()
    if not yaml_path.exists():
        raise FileNotFoundError('Invalid YAML path {}'.format(yaml_path))

    with open(yaml_path, 'r') as yf:
        return yaml.load(yf, Loader=yaml.SafeLoader) or {}


def main(args):
    parser = argparse.ArgumentParser('Clustering with KlustaKwik')
    parser.add_argument('target', help='Target path, either path containing tetrode files, or single tetrodeXX.mat')
    parser.add_argument('--KK', help='Path to KlustaKwik executable')
    parser.add_argument('--features', nargs='*', help='list of features to use for clustering')
    parser.add_argument('--config', help='Path to configuration file')
    parser.add_argument('--cluster', action='store_true', help='Directly run ')
    cli_args = parser.parse_args(args)

    # Load default configuration yaml file
    default_cfg_path = Path(pkg_resources.resource_filename(__name__, '../resources/cluster_defaults.yml')).resolve()
    if not default_cfg_path.exists():
        logging.error('Could not find default config file.')
        raise FileNotFoundError
    logger.debug('Loading default configuration')
    cfg = load_yaml(default_cfg_path)

    # Load local config file if it exists
    local_cfg_path = Path(pkg_resources.resource_filename(__name__, '../resources/cluster_defaults_local.yml')).resolve()
    if local_cfg_path.exists():
        logger.debug('Loading and updating with local configuration')
        local_cfg = load_yaml(local_cfg_path)
        cfg.update(local_cfg)

    # Load custom config path
    custom_cfg_path = Path(cli_args.config).resolve() if cli_args.config else None
    if custom_cfg_path:
        if custom_cfg_path.exists():
            logger.debug('Loading and updating with custom configuration')
            cfg.update(load_yaml(custom_cfg_path))
        else:
            raise FileNotFoundError(f"Could not load configuration file {custom_cfg_path}")

    # Load parameters from command line
    logger.debug('Parsing and updating configuration with CLI arguments')
    cfg.update(vars(cli_args))
    logger.debug(cfg)

    # try to find Klustakwik executable if necessary...
    if cli_args.KK is None:
        cli_args.KK = shutil.which('KlustaKwik') or shutil.which('klustakwik') or shutil.which('Klustakwik')
    if cli_args.KK is None and cli_args.cluster:
        raise FileNotFoundError('Could not find the KlustaKwik executable on the path, and none given.')

    # Find targets for operations
    target = Path(cli_args.target)
    if target.is_file() and target.exists():
        tetrode_files = [target]
        target = target.parent
        logger.debug('Using single file mode with {}'.format(target))
    else:
        tetrode_files = sorted(target.glob(cfg['TARGET_FILE_GLOB']))

    print(tetrode_files)


if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
