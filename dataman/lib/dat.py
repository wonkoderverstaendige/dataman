import sys
import os.path as op
from .tools import fext, path_content, fmt_time
import numpy as np
import logging

logger = logging.getLogger(__name__)

FMT_NAME = 'DAT'
ITEMSIZE = 2
AMPLITUDE_SCALE = 1 / 2 ** 10


def detect(base_path, pre_walk=None):
    """Checks for existence of a/multiple .dat file(s) at the target path.

    Args:
        base_path: Directory to search in.
        pre_walk: Tuple from previous path_content call (root, dirs, files)

    Returns:
        None if no data set found, else string
    """
    root, dirs, files = path_content(base_path) if pre_walk is None else pre_walk


    logger.debug('Detecting in dirs: {}, files: {}'.format(dirs, files))
    dat_files = [f for f in files if fext(f) == '.dat']
    logger.debug('# Dat files: {}'.format(len(dat_files)))
    if not len(dat_files):
        return None
    elif len(dat_files) == 1:
        return '{}-File'.format(FMT_NAME)
    else:
        return '{}x {}'.format(len(dat_files), FMT_NAME)


def config(base_path, *args, **kwargs):
    if not 'n_channels' in kwargs:
        raise BaseException('Unknown channel number, unable to unravel flat dat file.')

    return {'HEADER': {'sampling_rate': None,
                       'block_size': 1024,
                       'n_samples': op.getsize(base_path) / ITEMSIZE / kwargs['n_channels']},
            'DTYPE': None,
            'INFO': None,
            'SIGNALCHAIN': None,
            'FPGA_NODE': None,
            'AUDIO': None}


def fill_buffer(target, buffer, offset, count, *args, **kwargs):
    channels = kwargs['channels']
    byte_offset = offset * len(channels) * ITEMSIZE * 1024
    n_samples = count * len(channels)
    dtype = kwargs['dtype'] if 'dtype' in kwargs else 'int16'

    with open(target, 'rb') as dat_file:
        dat_file.seek(byte_offset)
        logger.debug('offset: {}, byte_offset: {}, count: {}'.format(offset, byte_offset, count))
        chunk = np.fromfile(dat_file, count=n_samples, dtype=dtype).reshape(-1, len(channels)).T.astype(
            'float32') * AMPLITUDE_SCALE
        buffer[:count] = chunk
