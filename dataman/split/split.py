import os
import numpy as np
from oio.util import run_prb, make_prm
from oio.formats import dat
import logging
from tqdm import tqdm
import sys
from contextlib import ExitStack
import math

logger = logging.getLogger(__name__)


def main(args):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Dat file')
    parser.add_argument('-o', '--out', help='Directory to store segments in', default='.')
    parser.add_argument('-c', '--clean', action='store_true', help='Remove the original dat file when successful')
    parser.add_argument('-C', '--channels', type=int, help='Number of channels in input file.')
    parser.add_argument('-d', '--dtype', default='int16')
    parser.add_argument('-p', '--prefix', default='{infile}_')

    grouping = parser.add_mutually_exclusive_group()
    grouping.add_argument('-l', '--layout', help='Path to probe file defining channel order')
    grouping.add_argument('-g', '--groups_of', type=int, help='Split into regular groups of n channels')

    cli_args = parser.parse_args(args)
    logger.debug('cli_args: {}'.format(cli_args))

    in_path = os.path.abspath(os.path.expanduser(cli_args.input))
    bp, ext = os.path.splitext(in_path)

    probe_file = cli_args.layout
    if not any([cli_args.layout, cli_args.groups_of]):
        if os.path.exists(bp + '.prb'):
            probe_file = bp + '.prb'
        else:
            logger.error('No information on how to split the channels. Either by groups_of, or with a prb file')
            sys.exit(1)

    if probe_file is not None:
        layout = run_prb(probe_file)
        channel_groups = layout['channel_groups']
        n_channels = sum([len(cg['channels']) for idx, cg in channel_groups.items()])
        logger.debug('{} channels from prb file'.format(n_channels))
    else:
        if cli_args.channels is None:
            logging.warning('No channel count given. Guessing...')
            n_channels = dat.guess_n_channels(in_path)
            logging.warning('Guessed there to be {} channels'.format(n_channels))
        else:
            n_channels = cli_args.channels
        assert not n_channels % cli_args.groups_of
        channel_groups = {cg: {'channels': list(range(cg * cli_args.groups_of, (cg + 1) * cli_args.groups_of))} for cg
                          in range(n_channels // cli_args.groups_of)}

    logging.debug('channel_groups: {}'.format(channel_groups))

    mm = np.memmap(in_path, dtype=cli_args.dtype, mode='r').reshape(-1, n_channels)

    indices = list(channel_groups.keys())
    batch_size = 1000000
    n_samples = mm.shape[0]
    pbar = tqdm(total=n_samples, unit_scale=True, unit='Samples')
    postfix = '{cg_id:0' + str(math.floor(math.log10(len(indices))) + 1) + 'd}.dat'

    with ExitStack() as stack:
        out_files = [stack.enter_context(open((cli_args.prefix + postfix).format(cg_id=cg_id, infile=bp), 'wb')) for
                     cg_id in range(len(indices))]
        samples_remaining = n_samples
        while samples_remaining > 0:
            pbar.update(batch_size)
            offset = n_samples - samples_remaining
            arr = mm[offset:offset + batch_size, :]
            for cg_id in range(len(indices)):
                arr.take(channel_groups[indices[cg_id]]['channels'], axis=1).tofile(out_files[cg_id])
            samples_remaining -= batch_size

        logger.debug('Writing .prm files')
        for outfile in out_files:
            print(outfile.name)
            make_prm(outfile.name, 'tetrode.prb')

    del mm



    try:
        if cli_args.clean:
            logger.warning('Deleting file {}'.format(in_path))
            os.remove(in_path)
    except PermissionError:
        logger.error("Couldn't clean up files. Sadface.")
