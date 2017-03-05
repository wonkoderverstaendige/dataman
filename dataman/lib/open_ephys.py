#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import xml.etree.ElementTree as ET
from .tools import fext, dir_content
import numpy as np
import re


SIZE_HEADER = 1024  # size of header in B
NUM_SAMPLES = 1024  # number of samples per record
SIZE_RECORD = 2070  # total size of record (2x1024 B samples + record header)
REC_MARKER = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 255], dtype=np.uint8)

# data type of .continuous open ephys 0.2x file format header
HEADER_DT = np.dtype([('Header', 'S%d' % SIZE_HEADER)])

# (2048 + 22) Byte = 2070 Byte total
# FIXME: The rec_mark comes after the samples. Currently data is read assuming full NUM_SAMPLE record!
DATA_DT = np.dtype([('timestamp', np.int64),            # 8 Byte
                    ('n_samples', np.uint16),           # 2 Byte
                    ('rec_num', np.uint16),             # 2 Byte
                    ('samples', ('>i2', NUM_SAMPLES)),  # 2 Byte each x 1024 typ.
                    ('rec_mark', (np.uint8, 10))])      # 10 Byte


def read_header(filename):
    """Return dict with .continuous file header content."""
    # TODO: Compare headers, should be identical except for channel

    # 1 kiB header string data type
    header = read_segment(filename, offset=0, count=1, dtype=HEADER_DT)

    # Stand back! I know regex!
    # Annoyingly, there is a newline character missing in the header (version/header_bytes)
    regex = "header\.([\d\w\.\s]{1,}).=.\'*([^\;\']{1,})\'*"
    header_str = str(header[0][0]).rstrip(' ')
    header_dict = {group[0]: group[1] for group in re.compile(regex).findall(header_str)}
    for key in ['bitVolts', 'sampleRate']:
        header_dict[key] = float(header_dict[key])
    for key in ['blockLength', 'bufferSize', 'header_bytes', 'channel']:
        header_dict[key] = int(header_dict[key] if not key == 'channel' else header_dict[key][2:])

    return header_dict


def read_segment(filename, offset, count, dtype):
    """Read segment of a file from [offset] for [count]x[dtype]"""
    with open(filename, 'rb') as fid:
        fid.seek(int(offset))
        segment = np.fromfile(fid, dtype=dtype, count=count)
    return segment


def read_record(filename, offset=0, count=30, dtype=DATA_DT):
    return read_segment(filename, offset=SIZE_HEADER+offset*SIZE_RECORD, count=count, dtype=dtype)['samples']\
        .ravel()\
        .astype(np.float32)/2**10


def detect(base=None, dirs=None, files=None):
    """Checks for existence of an open ephys formatted data set in the root directory.

    Args:
        base: Directory to search in.
        dirs: list of subdirectories in root. Will be scanned if not provided.
        files: List of files in the root directory. Will be scanned if not provided.

    Returns:
        None if no data set found, else a dict of configuration data from settings.xml
    """
    # TODO: Make all three optional and work with either
    if dirs is None or files is None:
        _, dirs, files = dir_content(base)

    for f in files:
        if fext(f) in ['.continuous']:
            # settings_xml = find_settings_xml(root)
            # if settings_xml is None:
            #     fv = None
            # else:
            fv = config(base)['INFO']['VERSION']
            return "OE_v{}".format(fv if fv else '???')
    else:
        return None

def _fpga_node(path):
    chain = config(path)['SIGNALCHAIN']
    nodes = [p['attrib']['NodeId'] for p in chain if p['type']=='PROCESSOR' and 'FPGA' in p['attrib']['name']]
    if len(nodes) == 1:
        return nodes[0]
    else:
        raise BaseException('Node ID not found in xml file {}'.format(path))


# def format_version(path):
#         settings = config(path)
#         return settings
#

def find_settings_xml(base, dirs=None, files=None):
    if dirs is None or files is None:
        _, dirs, files = dir_content(base)
    if "settings.xml" in files:
        return os.path.join(base, 'settings.xml')
    else:
        return None


def config(path):
    """Reads Open Ephys XML settings file and returns dictionary with relevant information.
        - Info field
            Dict(GUI version, date, OS, machine),
        - Signal chain
            List(Processor or Switch dict(name, nodeID). Signal chain is returned in ORDER OF OCCURRENCE
            in the xml file, this represents the order in the signal chain. The nodeID does NOT reflect this order, but
            but the order of assembling the chain.
        - Audio
            Int bufferSize
        - Header
            Dict(data from a single file header, i.e. sampling rate, blocks, etc.)

    Args:
        path: Path to settings.xml file

    Returns:
        Dict{INFO, SIGNALCHAIN, AUDIO, HEADER}
    """

    # Settings.xml file
    xml_path = find_settings_xml(path)
    root = ET.parse(xml_path).getroot()
    info = dict(
        VERSION = root.find('INFO/VERSION').text,
        DATE = root.find('INFO/DATE').text,
        OS = root.find('INFO/OS').text,
        MACHINE = root.find('INFO/VERSION').text
    )
    sc = root.find('SIGNALCHAIN')
    chain = [dict(type=e.tag, attrib=e.attrib) for e in sc.getchildren()]
    audio = root.find('AUDIO').attrib

    # Data file header (reliable sampling rate information)
    # FIXME: Make sure all headers agree...
    file_name = os.path.join(path, '106_CH1.continuous')
    header = read_header(file_name)
    fs = header['sampleRate']
    n_blocks = (os.path.getsize(file_name)-1024)/2070
    n_samples = n_blocks*1024

    # self.logger.info('Fs = {}kHz, {} blocks, {:.0f} samples, {:02.0f}min:{:02.0f}s'
    #                  .format(fs/1e3, n_blocks, n_samples,
    #                          math.floor(n_samples/fs/60),
    #                          math.floor(n_samples/fs%60)))
    header = dict(n_blocks=n_blocks,
                  block_size=NUM_SAMPLES,
                  n_samples=n_samples,
                  sampling_rate=fs)

    return dict(INFO=info,
                SIGNALCHAIN=chain,
                AUDIO=audio,
                HEADER=header)
