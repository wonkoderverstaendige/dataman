# generate .fet file from .fd files
#           OR
# if those features are missing, generate them with dm features!
# - generate/load features
# - run klustakwik
# - read clusters, build initial clustering report
import logging
import argparse
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Clustering with KlustaKwik')
    parser.add_argument('target', help='Target path, either path containing tetrode files, or single tetrodeXX.mat')
    parser.add_argument('--KK', help='Path to KlustaKwik executable')
    parser.add_argument('--features', nargs='*', help='list of features to use for clustering')
    cli_args = parser.parse_args()

    if cli_args.KK is None:
        cli_args.KK = shutil.which('KlustaKwik')

    if cli_args.KK is None:
        raise FileNotFoundError('Could not find the KlustaKwik executable on the path, and none given.')

    target = Path(cli_args.target)
    if target.is_file() and target.exists():
        tetrode_files = [target]
        target = target.parent
        logger.debug('Using single file mode with {}'.format(target))
    else:
        tetrode_files = sorted(target.glob('tetrode*.dat'))
