#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on Sep 23, 2015 18:53
@author: <'Ronny Eichler'> ronny.eichler@gmail.com

File reader
"""

import os
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

    # Add some additional information in there
    header_dict['file_bytes'] = os.path.getsize(filename)
    header_dict['numSamples'] = (header_dict['file_bytes']-SIZE_HEADER)/SIZE_RECORD*NUM_SAMPLES

    return header_dict


def read_segment(filename, offset, count, dtype):
    """Read segment of a file from [offset] for [count]x[dtype]"""
    with open(filename, 'rb') as fid:
        fid.seek(offset)
        segment = np.fromfile(fid, dtype=dtype, count=count)
    return segment


def read_record(filename, offset=0, count=30, dtype=DATA_DT):
    return read_segment(filename, offset=SIZE_HEADER+offset*SIZE_RECORD, count=count, dtype=dtype)['samples']\
        .ravel()\
        .astype(np.float32)/2**10

# def reader(filename, buf):
#     """
#     Reader for a single .continuous file. Writes as many (complete, i.e. NUM_SAMPLES) records into given
#     the buffer as can fit.
#     :param filename: File name of the input .continuous file.
#     :param buf: Designated column of a numpy array used for temporary storage/stacking of multiple channel data.
#     :return: Dictionary of all headers read and stored in buffer.
#     """
#     # TODO: Allow sending new index to generator
#     # TODO: Check record size for completion
#     with open(filename, 'rb') as fid:
#         yield np.fromfile(fid, HEADER_DT, 1)
#         while True:
#             data = np.fromfile(fid, DATA_DT, len(buf)/NUM_SAMPLES)
#             buf[:len(data)*NUM_SAMPLES] = data['samples'].ravel()
#             yield {idx: data[idx] for idx in data.dtype.names if idx != 'samples'} if len(data) else None


if __name__ == "__main__":
    print read_header('data/2014-10-30_16-07-29/106_CH1.continuous')
    print read_segment('data/2014-10-30_16-07-29/106_CH1.continuous', offset=SIZE_HEADER)['samples'].ravel()[0]





