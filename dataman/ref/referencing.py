from os import path as op, remove
from shutil import copyfile
import numpy as np
from oio.util import get_batch_size, run_prb, flat_channel_list, has_prb
from oio.formats import dat
import logging
from tqdm import trange

logger = logging.getLogger(__name__)


def ref(dat_path, ref_path=None, keep=False, inplace=False, *args, **kwargs):
    dat_path = op.abspath(op.expanduser(dat_path))
    if ref_path is None:
        logger.info('Creating reference...')
        ref_path = make_ref_file(dat_path, *args, **kwargs)
        logger.debug('Reference file at {}'.format(ref_path))

    assert ref_path

    logger.info('Reference subtraction')
    subtract_reference(dat_path, ref_path, inplace=inplace, *args, **kwargs)

    # Copy the prb file
    if not inplace:
        logger.warning('Copying probe file to follow referenced data.')

    if not keep:
        logger.warning('Not keeping reference file. Deleting it...')
        remove(ref_path)


def subtract_reference(dat_path, ref_path, precision='single', inplace=False,
                       n_channels=64, ch_idx_bad=None, zero_bad_channels=False, *args, **kwargs):
    # if inplace, just overwrite, in_file, else, make copy of in_file
    # FIXME: Also memmap the reference file? Should be small even for long recordings...
    # FIXME: If not inplace, the file should be opened read only!
    # TODO: Allow immediate zero-ing of bad channels
    logger.debug('Subtracting {} from {}'.format(ref_path, dat_path))
    logger.debug('Precision: {}, inplace={} with {} channels'.format(precision, inplace, n_channels))
    logger.debug('Opening files, dat: {}; ref: {}'.format(dat_path, ref_path))
    with open(dat_path, 'r+b') as dat, open(ref_path, 'rb') as mu:
        dat_arr = np.memmap(dat, dtype='int16').reshape(-1, n_channels)
        mu_arr = np.fromfile(mu, dtype=precision).reshape(-1, 1)
        assert (dat_arr.shape[0] == mu_arr.shape[0])

    batch_size, n_batches, batch_size_last = get_batch_size(dat_arr)
    fname, ext = op.splitext(dat_path)

    logger.debug('Bad channels: {}, zeroing: {}'.format(ch_idx_bad, zero_bad_channels))

    try:
        if inplace:
            out_arr = dat_arr
        else:
            out_arr = np.memmap(fname + '_meanref' + ext, mode='w+', dtype=dat_arr.dtype, shape=dat_arr.shape)

        for bn in trange(n_batches + 1):
            bs = batch_size if bn < n_batches else batch_size_last
            if inplace:
                out_arr[bn * bs:(bn + 1) * bs, :] -= mu_arr[bn * bs:(bn + 1) * bs].astype(out_arr.dtype)
            else:
                out_arr[bn * bs:(bn + 1) * bs, :] = dat_arr[bn * bs:(bn + 1) * bs, :] - mu_arr[bn * bs:(bn + 1) * bs]

            if zero_bad_channels and ch_idx_bad is not None:
                out_arr[bn * bs:(bn + 1) * bs, ch_idx_bad] = 0

    except BaseException as e:
        print(e)


def make_ref_file(dat_path, n_channels, ref_out_fname=None, *args, **kwargs):
    # calculate mean over given array.
    if ref_out_fname is None:
        fname, ext = op.splitext(dat_path)
        ref_out_fname = fname + '_reference' + ext

    with open(dat_path, 'rb') as dat, open(ref_out_fname, 'wb+') as mu:
        dat_arr = np.memmap(dat, mode='r', dtype='int16').reshape(-1, n_channels)
        _batch_reference(dat_arr, mu, *args, **kwargs)
    return ref_out_fname


def _batch_reference(arr_like, out_file, ch_idx_good=None, ch_idx_bad=None, precision='float32', *args, **kwargs):
    # out_file is file pointer!
    batch_size, n_batches, batch_size_last = get_batch_size(arr_like)
    assert not (ch_idx_good is not None and ch_idx_bad is not None)
    if ch_idx_bad is not None:
        ch_idx_good = [c for c in range(arr_like.shape[1]) if c not in ch_idx_bad]
    if ch_idx_good is not None:
        n_channels = len(ch_idx_good)
    else:
        n_channels = arr_like.shape[1]

    logger.debug(
        'Reference will to be created at {} from {} channels'.format(out_file.name, n_channels))

    for bn in trange(n_batches + 1):
        bs = batch_size if bn < n_batches else batch_size_last
        if ch_idx_good is None:
            batch = arr_like[bn * bs:(bn + 1) * bs, :]
        else:
            batch = arr_like[bn * bs:(bn + 1) * bs, :].take(ch_idx_good, axis=1)
        np.mean(batch, axis=1, dtype=precision).tofile(out_file)


def copy_as(src, dst):
    logger.info('Copying file {} to {}.'.format(src, dst))
    copyfile(src, dst)


def main(args):
    # TODO: Mutually exclusive bad/good channels
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Dat file')
    parser.add_argument('-o', '--out', help='Directory to store reference file in', default='.')
    parser.add_argument('-r', '--reference', help='Path to reference file, if already at hand.')
    parser.add_argument('-b', '--bad_channels', type=int, nargs='+', help='Dead channel indices')
    parser.add_argument('-g', '--good_channels', type=int, nargs='+', help='Indices of channels to include')
    parser.add_argument('-C', '--channels', type=int, help='Number of channels in input file.')
    parser.add_argument('-Z', '--zero_bad_channels', action='store_true', help='Set bad channels to zero.')
    parser.add_argument('-i', '--inplace', action='store_true', help='Subtract reference in place.')
    parser.add_argument('-m', '--make-only', action='store_true', help='Only create the reference file.')
    parser.add_argument('-l', '--layout', help='Path to probe file defining channel order')
    parser.add_argument('-k', '--keep', action='store_true', help='Keep intermediate reference file')
    cli_args = parser.parse_args(args)

    n_channels = cli_args.channels if 'channels' in cli_args else None
    cfg = dat.metadata(cli_args.input, n_channels=n_channels)
    if n_channels is None:
        n_channels = cfg['CHANNELS']['n_channels']
    logger.debug(cfg)

    # FIXME: Assumes "pre-ordered" channels, i.e. 0:n_channels
    probe_file = cli_args.layout if 'layout' in cli_args else None

    if probe_file is None and has_prb(cli_args.input):
        probe_file = has_prb(cli_args.input)
        logger.warning('No probe file given, but .prb file found. Using {}'.format(probe_file))

    if probe_file is not None:
        layout = run_prb(probe_file)

        channels, bad_channels = flat_channel_list(layout)[:n_channels]
        print(bad_channels)
    else:
        channels = None
        bad_channels = None

    logger.debug('Good: {}, bad: {}'.format(channels, bad_channels))

    ref(cli_args.input,
        ref_path=cli_args.reference,
        out_dir=cli_args.out,
        n_channels=n_channels,
        ch_idx_good=cli_args.good_channels,
        ch_idx_bad=bad_channels,
        zero_bad_channels=cli_args.zero_bad_channels,
        make_only=cli_args.make_only,
        inplace=cli_args.inplace,
        keep=cli_args.keep)
