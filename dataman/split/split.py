import os
import numpy as np
from oio.util import run_prb
from oio.formats import dat
import logging
from tqdm import trange
import sys

logger = logging.getLogger(__name__)


def main(args):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Dat file')
    parser.add_argument('-o', '--out', help='Directory to store segments in', default='.')
    parser.add_argument('-c', '--clean', action='store_true', help='Remove the original dat file when successful')
    parser.add_argument('-C', '--channels', type=int, help='Number of channels in input file.')
    parser.add_argument('-d', '--dtype', default='int16')

    grouping = parser.add_mutually_exclusive_group()
    grouping.add_argument('-l', '--layout', help='Path to probe file defining channel order')
    grouping.add_argument('-g', '--groups_of', type=int, help='Split into regular groups of n channels')

    cli_args = parser.parse_args(args)
    logger.debug('cli_args: {}'.format(cli_args))

    in_file = os.path.abspath(os.path.expanduser(cli_args.input))

    if not any([cli_args.layout, cli_args.groups_of]):
        logger.error('No information on how to split the channels. Either by groups_of, or with a prb file')
        sys.exit(1)

    if cli_args.layout is not None:
        layout = run_prb(cli_args.layout)
        channel_groups = layout['channel_groups']
        n_channels = sum([len(cg['channels']) for id, cg in channel_groups.items()])
        logger.debug('{} channels from prb file'.format(n_channels))
    else:
        if cli_args.channels is None:
            logging.warning('No channel count given. Guessing...')
            n_channels = dat.guess_n_channels(in_file)
            logging.warning('Guessed there to be {} channels'.format(n_channels))
        else:
            n_channels = cli_args.channels
        assert not n_channels % cli_args.groups_of
        channel_groups = {cg: {'channels': list(range(cg * cli_args.groups_of, (cg + 1) * cli_args.groups_of))} for cg in
                          range(n_channels // cli_args.groups_of)}

    logging.debug('channel_groups: {}'.format(channel_groups))

    mm = np.memmap(in_file, dtype=cli_args.dtype, mode='r').reshape(-1, n_channels)

    indices = list(channel_groups.keys())
    for id in trange(len(indices)):
        arr = mm.take(channel_groups[indices[id]]['channels'], axis=1)
        arr.tofile('test_{}.dat'.format(indices[id]))

    if cli_args.clean:
        logger.warning('Deleting file {}'.format(in_file))
        os.remove(in_file)
