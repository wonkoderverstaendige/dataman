#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
import os.path as op
import pprint
import re
from collections import Counter

from scipy import signal
from termcolor import colored

from dataman.formats import get_valid_formats
from dataman.lib.constants import DEFAULT_MEMORY_LIMIT_MB

ansi_escape = re.compile(r'\x1b[^m]*m')
logger = logging.getLogger(__name__)


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a


def get_batch_size(arr, ram_limit=DEFAULT_MEMORY_LIMIT_MB):
    """Get batch size for an array given memory limit per batch"""
    batch_size = int(ram_limit * 1e6 / arr.shape[1] / arr.dtype.itemsize)
    return batch_size


def get_batch_limits(length, batch_size):
    starts = [bc * batch_size for bc in range(length // batch_size + 1)]
    ends = [start + batch_size for start in starts[:-1]]
    ends.append(length)
    # if length - (length // batch_size) * batch_size:
    #     batches.append(length - length % batch_size)
    # return batches
    return list(zip(starts, ends))


def detect_format(path, return_singlular=True):
    """Check if/what known data formats are present at the given path and return the module needed to interact with it.

    Args:
        path: Target path
        return_singlular: bool, indicate whether to allow and return only a single format

    Returns:
        Single data format object, or list of formats
    """

    formats = [fmt for fmt in get_valid_formats() if fmt.detect(path)]
    if return_singlular:
        if len(formats) == 1:
            return formats[0]
        else:
            raise ValueError('More than one format detected at target ({}): {}'.format(path, formats))
    else:
        return formats
    # formats = [f for f in [fmt.detect(path) for fmt in get_valid_formats()] if f is not None]
    # if len(formats) == 1:
    #     fmt = formats[0]
    #     if 'DAT' in fmt:
    #         if fmt == 'DAT-File':
    #             return dat
    #     else:
    #         if 'kwik' in fmt:
    #             return kwik
    #         else:
    #             return open_ephys
    # logger.info('Detected format(s) {} not valid.'.format(formats))


def run_prb(path):
    """Execute the .prb probe file and import resulting locals return results as dict.
    Args:
        path: file path to probe file with layout as per klusta probe file specs.

    Returns: Dictionary of channel groups with channel list, geometry and connectivity graph.
    """

    if path is None:
        return

    path = op.realpath(op.expanduser(path))
    assert op.exists(path)
    with open(path, 'r') as prb:
        layout = prb.read()

    metadata = {}
    exec(layout, {}, metadata)
    metadata = {k.lower(): v for (k, v) in metadata.items()}
    return metadata


def write_prb(prb_path, channel_groups, dead_channels=None):
    """Write a .prb file with given channel group dictionary and list of dead channels
    """
    dead_channels = [] if dead_channels is None else dead_channels
    with open(prb_path, 'w') as prb_file:
        prb_file.write('dead_channels = {}\n'.format(pprint.pformat(dead_channels)))
        prb_file.write('channel_groups = {}\n'.format(pprint.pformat(channel_groups)))


def flat_channel_list(prb):
    """Flat lists of all channels and bad channels given a probe dictionary.

    Args:
        prb: Path to probe file.

    Returns:
        Tuple of lists of (all, bad) channels.
    """

    channels = sum([prb['channel_groups'][cg]['channels'] for cg in sorted(prb['channel_groups'])], [])
    dead_channels = prb['dead_channels'] if 'dead_channels' in prb else []

    return channels, dead_channels


def monotonic_prb(prb):
    """Return probe file dict with monotonically increasing channel group and channel numbers."""

    # FIXME: Should otherwise copy any other fields over (unknown fields warning)
    # FIXME: Correct ref like dc to indices
    chan_n = 0
    groups = prb['channel_groups']
    monotonic = {}
    for n, chg in enumerate(groups.keys()):
        monotonic[n] = {'channels': list(range(chan_n, chan_n + len(groups[chg]['channels'])))}
        chan_n += len(groups[chg]['channels'])

    # correct bad channel indices
    if 'dead_channels' in prb.keys():
        fcl, fbc = flat_channel_list(prb)
        dead_channels = sorted([fcl.index(bc) for bc in fbc])
    else:
        dead_channels = []
    return monotonic, dead_channels


def make_prb():
    raise NotImplemented


def make_prm():
    raise NotImplemented
# def make_prm(dat_path, prb_path, n_channels=4):
#     prm_in = pkgr.resource_string('config', 'default.prm').decode()
#     #
#     # with open('default.prm', 'r') as prm_default:
#     #     template = prm_default.read()
#
#     base_name, _ = op.splitext(dat_path)
#     with open(base_name + '.prm', 'w') as prm_out:
#         prm_out.write(prm_in.format(experiment_name=base_name,
#                                     probe_file=prb_path,
#                                     n_channels=4))


def has_prb(filepath):
    """Check if file at path has a an accompanying .prb file with the same basename.

    Args:
        filepath: Path to file of interest

    Returns:
        Path to .prb file if exists, else None
    """
    # TODO: Use pathlib
    base_path, _ = op.splitext(op.abspath(op.expanduser(filepath)))
    probe_path = base_path + '.prb'
    if op.exists(probe_path) and op.isfile(probe_path):
        return probe_path


def channel_ranges(channel_list):
    """List of channels in to ranges of consecutive channels.

    Args:
        channel_list: list of channel numbers (ints)

    Returns:
        List of list of channels grouped into consecutive sequences.

    Example: channel_ranges([1, 3, 4, 5]) -> [[1], [3, 4, 5]]
    """

    duplicates = [c for c, n in Counter(channel_list).items() if n > 1]
    if len(duplicates):
        logger.warning("Channel(s) {} listed more than once".format(duplicates))

    ranges = [[]]
    for channel in channel_list:
        if len(ranges[-1]) and abs(channel - ranges[-1][-1]) != 1:
            ranges.append([])
        ranges[-1].append(channel)
    return ranges


def fmt_channel_ranges(channels, shorten_seq=5, rs="tm", c_sep="_", zp=2):
    """String of channel numbers separated with delimiters with consecutive channels
    are shortened when sequence length above threshold.

    Args:
        channels: list of channels
        shorten_seq: number of consecutive channels to be shortened. (default: 5)
        rs: range delimiter (default: 'tm')
        c_sep: channel delimiter (default: '_')
        zp: zero pad channel numbers (default: 2)

    Returns: String of channels in order they appeared in the list of channels.

    Example: fmt_channel_ranges([[1], [3], [5, 6, 7, 8, 9, 10]]) -> 01_03_05tm10
    """
    c_ranges = channel_ranges(channels)
    range_strings = [c_sep.join(["{c:0{zp}}".format(c=c, zp=zp) for c in c_seq])
                     if len(c_seq) < shorten_seq
                     else "{start:0{zp}}{rs}{end:0{zp}}".format(start=c_seq[0], end=c_seq[-1], rs=rs, zp=zp)
                     for c_seq in c_ranges]
    return c_sep.join(range_strings)


def fmt_size(num, unit='B', si=True, sep=' ', col=False, pad=0):
    colors = {"k": "blue", "M": "green", "G": "red", "T": "cyan",
              "Ki": "blue", "Mi": "green", "Gi": "red", "Ti": "cyan"}
    if si:
        prefixes = ['', 'k', 'M', 'G', 'T', 'P', 'E']
    else:
        prefixes = ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei']
    
    divisor = 1000 if si else 1024
    for prefix in prefixes:
        if abs(num) < divisor:
            if prefix:
                prefix = colored(prefix, colors[prefix]) if col else prefix
                return "{:5.1f}{}{}{}".format(num, sep, prefix, unit, pad=pad-6)
            else:
                return "{:5.0f}{}{}{} ".format(num, sep, prefix, unit, pad=pad-6)
        num /= divisor


def get_needed_channels(cli_args=None):
    """Gets a list of channels that are needed in order to process a given channel.
    """
    if cli_args is None:
        import argparse
        parser = argparse.ArgumentParser(description=get_needed_channels.__doc__)
        parser.add_argument('probe_file', nargs=1,
                            help="""Phe probe file to be used""")
        parser.add_argument("groups", nargs='+',
                            help="""A list of groups""")
        parser.add_argument("-f", "--filenames", action='store_true',
                            help="""Returns a list of open-ephys continuous filenames, instead of a list of channel
                                    numbers""")
        parser.add_argument("-n", "--node", type=int,
                            help="""A node number for the filenames (default 100)""")
        parser.add_argument("--zerobased", action='store_true',
                            help="Use klusta zero-based convention instead of open-ephys 1-based one")
        cli_args = parser.parse_args()

    probe_file = cli_args.probe_file[0]
    groups = [int(g) for g in cli_args.groups]

    do_filenames = False
    if cli_args.filenames:
        do_filenames = True

    if cli_args.node:
        node = cli_args.node
        do_filenames = True
    else:
        node = 100

    zero_based = False
    if cli_args.zerobased:
        zero_based = True

    layout = run_prb(probe_file)

    channels = []
    for g in groups:
        channels.extend(layout['channel_groups'][g]['channels'])
        if 'reference' in layout['channel_groups']:
            channels.extend(layout['channel_groups'][g]['reference'])

    if 'ref_a' in layout:
        channels.extend(layout['ref_a'])

    if 'ref_b' in layout:
        channels.extend(layout['ref_b'])

    if not zero_based:
        channels = [c + 1 for c in channels]
    channels = set(channels)

    if do_filenames:
        fnames = [str(node) + '_CH' + str(c) + '.continuous' for c in channels]
        print('\n'.join(fnames))
    else:
        print(' '.join(map(str, channels)))


def fmt_time(s, minimal=True, millis=True, delim=' '):
    """
    Args:
        s: time in seconds (float for fractional)
        minimal: Flag, if true, only return strings for times > 0, leave rest outs
    Returns: String formatted 99h 59min 59.9s, where elements < 1 are left out optionally.
    """
    sec_fmt = '{s:02.3f}s' if millis else '{s:02.0f}s'
    minutes_fmt = '{m:02d}min'
    hours_fmt = '{h:02d}h '

    ms = s-int(s)
    s = int(s)
    s_str = sec_fmt.format(s=s+ms)
    if s < 60 and minimal:
        return s_str

    m, s = divmod(s, 60)
    s_str = sec_fmt.format(s=s + ms)
    m_str = minutes_fmt.format(m=m)
    if m < 60 and minimal:
        return delim.join([m_str, s_str])

    h, m = divmod(m, 60)
    m_str = minutes_fmt.format(m=m)
    h_str = hours_fmt.format(h=h)
    return delim.join([h_str, m_str, s_str])
    # return " {m:02d}min {s:02.3f}s".format(h=h, m=m, s=s+ms)


def fext(fname):
    """Grabs the file extension of a file.

    Args:
        fname: File name.

    Returns:
        String with file extension. Empty string, if file has no extensions.

    Raises:
        IOError if file does not exist or can not be accessed.
    """
    return os.path.splitext(fname)[1]


def full_path(path):
    """Return full path of a potentially relative path, including ~ expansion.

    Args:
        path

    Returns:
        Absolute(Expanduser(Path))
    """
    return os.path.abspath(os.path.expanduser(path))


def path_content(path):
    """Gathers root and first level content of a directory.

    Args:
        path: Relative or absolute path to a directory.

    Returns:
        A tuple containing the root path, the directories and the files
        contained in the root directory.

        (path, dir_names, file_names)
    """
    path = full_path(path)
    assert(os.path.exists(path))
    if os.path.isdir(path):
        return next(os.walk(path))
    else:
        return os.path.basename(path), [], [path]


def dir_size(path):
    """Calculate size of directory including all subdirectories and files

    Args:
        path: Relative or absolute path.

    Returns:
        Integer value of size in Bytes.
    """
    logger.debug('dir_size path: {}'.format(path))
    assert os.path.exists(path)
    if not os.path.isdir(path):
        return os.path.getsize(path)

    total_size = 0
    for root, dirs, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            try:
                total_size += os.path.getsize(fp)
            except OSError:
                # symbolic links cause issues
                pass
    return total_size


def terminal_size():
    """Get size of currently used terminal. In many cases this is inaccurate.

    Returns:
        Tuple of width, height.

    Raises:
        Unknown error when not run from a terminal.
    """
    # return map(int, os.popen('stty size', 'r').read().split())
    # Python 3.3+
    ts = os.get_terminal_size()
    return ts.lines, ts.columns


def find_getch():
    """Helper to wait for a single character press, instead of having to use raw_input() requiring Enter
    to be pressed. Should work on all OS.

    Returns:
        Function that works as blocking single character input without prompt.
    """
    # FIXME: Find where I took this piece of code from... and attribute. SO perhaps?
    try:
        import termios
    except ImportError:
        # Non-POSIX. Return msvcrt's (Windows') getch.
        import msvcrt
        return msvcrt.getch

    # POSIX system. Create and return a getch that manipulates the tty.
    import sys
    import tty

    def _getch():
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

    return _getch


def strip_ansi(string):
    """Remove the ANSI codes (e.g. color and additional formatting) from a string.

    Args:
        string: A string potentially containing ANSI escape codes.

    Returns:
        String with ANSI escape codes removed.
    """
    return ansi_escape.sub('', string)


if __name__ == "__main__":
    pass
