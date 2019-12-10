import logging
import math
import os
import sys
from contextlib import ExitStack
from pathlib import Path

import numpy as np
from tqdm import tqdm

from dataman.formats import dat
from dataman.lib.util import run_prb, write_prb

logger = logging.getLogger(__name__)


def main(args):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Dat file')
    parser.add_argument('-o', '--out', help='Directory to store segments in', default='.')
    parser.add_argument('-c', '--clean', action='store_true', help='Remove the original dat file when successful')
    parser.add_argument('-C', '--channels', type=int, help='Number of channels in input file.')
    parser.add_argument('-d', '--dtype', default='int16')
    parser.add_argument('-p', '--prefix', default='tetrode',
                        help='Prefix to output file name. Default: "tetrode"')  # '{infile}_'
    parser.add_argument('--keep_dead', help='Do not skip tetrodes with all-dead channels', action='store_true')

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
        dead_channels = layout['dead_channels'] if 'dead_channels' in layout else []
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
        dead_channels = []

    logging.debug('channel_groups: {}'.format(channel_groups))

    mm = np.memmap(in_path, dtype=cli_args.dtype, mode='r').reshape(-1, n_channels)

    # Select valid channel groups, skip group with all-dead channels
    indices = []
    for cg in channel_groups.keys():
        channels = channel_groups[cg]['channels']
        dead = [ch in dead_channels for ch in channels]
        if all(dead):
            logger.warning(f'Skipping tetrode {cg} because all channels are dead. Use --keep_dead to not skip.')
            continue
        indices.append(cg)

    # # Create per-tetrode probe file
    # # FIXME: Dead channels are big mess
    # with open(op.join(out_path, output_basename + '.prb'), 'w') as prb_out:
    #     if cli_args.split_groups or (layout is None):
    #         # One prb file per channel group
    #         ch_out = channel_group['channels']
    #         cg_out = {0: {'channels': list(range(len(ch_out)))}}
    #         dead_channels = sorted([ch_out.index(dc) for dc in dead_channels if dc in ch_out])
    #
    #     else:
    #         # Same channel groups, but with flat numbering
    #         cg_out, dead_channels = util.monotonic_prb(layout)
    #     prb_out.write('dead_channels = {}\n'.format(pprint.pformat(dead_channels)))
    #     prb_out.write('channel_groups = {}'.format(pprint.pformat(cg_out)))

    batch_size = 1_000_000
    n_samples = mm.shape[0]
    pbar = tqdm(total=n_samples, unit_scale=True, unit='Samples')
    postfix = '{cg_id:0' + str(math.floor(math.log10(max(indices))) + 1) + 'd}.dat'

    with ExitStack() as stack:
        out_files = {}
        for cg_id in indices:
            dat_path = Path((cli_args.prefix + postfix).format(cg_id=cg_id, infile=bp))
            prb_path = dat_path.with_suffix('.prb')

            # Create per-tetrode probe file
            ch_out = channel_groups[cg_id]['channels']
            cg_out = {0: {'channels': list(range(len(ch_out)))}}
            dead_ch = sorted([ch_out.index(dc) for dc in dead_channels if dc in ch_out])
            write_prb(prb_path, cg_out, dead_ch)

            # Create file object for .dat file and append to exit stack for clean shutdown
            of = open(dat_path, 'wb')
            out_files[cg_id] = stack.enter_context(of)

        samples_remaining = n_samples
        while samples_remaining > 0:
            pbar.update(batch_size)
            offset = n_samples - samples_remaining
            arr = mm[offset:offset + batch_size, :]
            for cg_id in out_files.keys():
                arr.take(channel_groups[cg_id]['channels'], axis=1).tofile(out_files[cg_id])
            samples_remaining -= batch_size

    del mm

    try:
        if cli_args.clean:
            logger.warning('Deleting file {}'.format(in_path))
            os.remove(in_path)
    except PermissionError:
        logger.error("Couldn't clean up files. Sadface.")
