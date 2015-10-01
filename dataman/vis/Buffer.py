#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Mutable buffer/array that can be shared between multiple processes.
Inspired by https://github.com/belevtsoff/rdaclient.py

* `Buffer`: The buffer
* `datatypes`: supported datatypes
* `BufferHeader`: header structure containing metadata
* `BufferError`: error definitions

"""

from multiprocessing import Array
import ctypes as c
import logging

import numpy as np


class Buffer(object):
    """
    One-dimensional buffer with homogenous elements.

    The buffer can be used simultaneously by multiple processes, because
    both data and metadata are stored in a single sharedctypes byte array.

    First, the buffer object is created and initialized in one of the
    processes. Second, its raw array is shared with others. Third, those
    processes create their own Buffer objects and initialize them so that
    all point to the same shared raw array.
    """

    def __init__(self):
        self.logger = logging.getLogger("Buffer")
        self.__initialized = False

    def __str__(self):
        return self.__buf[:self.buf_size].__str__() + '\n'

    def __getattr__(self, item):
        """Overload to prevent access to the buffer attributes before
        initialization is complete.
        """
        if self.__initialized:
            return object.__getattribute__(self, item)
        else:
            raise BufferError(1)

    # -------------------------------------------------------------------------
    # PROPERTIES

    # read only attributes
    is_initialized = property(lambda self: self.__initialized, None, None,
                              'Indicates whether the buffer is initialized, read-only (bool)')
    raw = property(lambda self: self.__raw, None, None,
                   'Raw buffer array, read-only (sharedctypes, char)')
    n_channels = property(lambda self: self.__n_channels, None, None,
                          'Dimensionality of array in channels, read-only (int)')
    buf_samples = property(lambda self: self.__buf_samples, None, None,
                         'Dimensionality of array in samples, read-only (int)')
    buf_size = property(lambda self: self.__buf_size, None, None,
                        'Buffer size, read only (int)')
    position = property(lambda self: self.__position, None, None,
                        'Index of zero-th column in data samples, read-only (int)')
    np_type = property(lambda self: self.__np_type, None, None,
                       'The type of the data in the buffer, read-only (string)')

    # -------------------------------------------------------------------------

    def initialize(self, n_channels, buf_size_samples, np_type='float32'):
        """Initializes the buffer with a new array."""

        # check parameters
        if n_channels < 1 or buf_size_samples < 1:
            self.logger.error('n_channels and buf_size_samples must be positive integers')
            raise BufferError(1)

        size_bytes = c.sizeof(BufferHeader) + buf_size_samples * n_channels * np.dtype(np_type).itemsize
        raw = Array('c', size_bytes)
        hdr = BufferHeader.from_buffer(raw.get_obj())

        hdr.dataType = DataTypes.get_code(np_type)
        hdr.nChannels = n_channels
        hdr.bufSizeSamples = buf_size_samples
        hdr.bufSizeBytes = size_bytes - c.sizeof(BufferHeader)
        hdr.position = 0

        self.initialize_from_raw(raw.get_obj())

    def initialize_from_raw(self, raw):
        """Initiates the buffer with the compatible external raw array.
        All the metadata will be read from the header region of the array.
        """
        self.__initialized = True
        hdr = BufferHeader.from_buffer(raw)

        # data type of corresponding numpy array
        np_type = DataTypes.get_type(hdr.dataType)

        buf_offset = c.sizeof(hdr)
        buf_flat_size = hdr.bufSizeBytes / np.dtype(np_type).itemsize

        # create numpy view object pointing to the raw array
        self.__raw = raw
        self.__hdr = hdr
        self.__buf = np.frombuffer(raw, np_type, buf_flat_size, buf_offset) \
            .reshape((-1, hdr.bufSizeSamples))
        self.__buf_size = len(self.__buf)

        # helper variables
        self.__n_channels = hdr.nChannels
        self.__buf_samples = hdr.bufSizeSamples
        self.__np_type = np_type

    def __write_buffer(self, data, start, channel=None):
        """Writes data to buffer."""
        end = start + data.shape[1]
        if end > self.__buf_samples:
            raise BufferError("Attempting to write data beyond buffer size!")

        if channel is None:
            self.__buf[:, start:end] = data
        else:
            self.__buf[channel, start:end] = data

    def __update_buffer(self, data, position, channel=None):
        # find first index, calculate difference
        # if delta position > buffer size, refresh whole buffer, start = 0
        # if delta position < buffer size, roll array by delta, append/prepend data
        #   by overwriting the old section
        pass

    def __read_buffer(self, start, end=-1):
        """Reads data from buffer, returning view into numpy array"""
        # av_error = self.check_availablility(start, end)
        # if not av_error:
        #     return self.__buf[:, start:end]
        # else:
        #     raise BufferError(av_error)
        return self.__buf[:, start:end]

    def get_data(self, start, end=-1, wprotect=True):
        data = self.__read_buffer(start, end)
        data.setflags(write=not wprotect)
        return data

    def put_data(self, data, start=0):
        """Put data for all channels at once into the buffer."""
        if data.shape[0] != self.n_channels:
            raise BufferError(4)
        self.__write_buffer(data, start)

    def put_channel_data(self, data, channel, start=0):
        """Put data for single channel into the buffer."""
        data.shape = (1, len(data))
        if channel >= self.n_channels or channel < 0:
            raise BufferError(4)
        self.__write_buffer(data, start, channel=channel)

    # def check_availablility(self, start, end):
    #     """Checks whether the requested data samples are available.
    #
    #     Parameters
    #     ----------
    #     start : int
    #         first sample index (included)
    #     end : int
    #         last samples index (excluded)
    #
    #     Returns
    #     -------
    #     0
    #         if the data is available and already in the buffer
    #     1
    #         if the data is available but needs to be read in
    #     2
    #         if data is partially unavailable
    #     3
    #         if data is completely unavailable
    #     """
    #     if self.position <= start and self.position+self.buf_samples < end:
    #         return 0
    #     elif start > self.__buf_samples:
    #         return
        # if sampleStart < 0 or sampleEnd <= 0:
        #     return 5
        # if sampleEnd > self.nSamplesWritten:
        #     return 3  # data is not ready
        # if (self.nSamplesWritten - sampleStart) > self.bufSize:
        #     return 2  # data is already erased
        #
        # return 0


class DataTypes():
    """A helper class to interpret the type code read from buffer header.
    To add new supported data types, add them to the 'type' dictionary
    """
    types = {0: 'float32',
             1: 'int16'}

    @classmethod
    def get_code(cls, nd_type):
        """Gets buffer type code given numpy data type

        Parameters
        ----------
        nd_type : string
            numpy datatype (e.g. 'float32')
        """
        idx = cls.types.values().index(nd_type)
        return cls.types.keys()[idx]

    @classmethod
    def get_type(cls, code):
        """Gets numpy data type given a buffer type code

        Parameters
        ----------
        code : int
            type code (e.g. 0)
        """
        return cls.types[code]


class BufferHeader(c.Structure):
    """A ctypes structure describing the buffer header

    Attributes
    ----------
    bufSizeBytes : c_ulong
        size of the buffer in bytes, excluding header and pocket
    dataType : c_uint
        typecode of the data stored in the buffer
    nChannels : c_ulong
        sample dimensionality
    nSamples : c_ulong
        size of the buffer in samples
    position : c_ulong
        position in the data in samples
    """
    _pack_ = 1
    _fields_ = [('dataType', c.c_uint),
                ('nChannels', c.c_ulong),
                ('bufSizeSamples', c.c_ulong),
                ('bufSizeBytes', c.c_ulong),
                ('position', c.c_ulong)]


class BufferError(Exception):
    """Represents different types of buffer errors"""

    def __init__(self, code):
        """Initializes a BufferError with given error code

        Parameters
        ----------
        code : int
            error code
        """
        self.code = code

    def __str__(self):
        """Prints the error"""
        if self.code == 1:
            return 'buffer is not initialized (error %s)' % repr(self.code)
        elif self.code in [2, 3]:
            return 'unable to get indices (error %s)' % repr(self.code)
        elif self.code == 4:
            return 'writing incompatible data (error %s)' % repr(self.code)
        elif self.code == 5:
            return 'negative index (error %s)' % repr(self.code)
        else:
            return '(error %s)' % repr(self.code)


if __name__ == '__main__':
    buf1 = Buffer()
    buf2 = Buffer()

    buf1.initialize(2, 15)
    buf2.initialize_from_raw(buf1.raw)

    buf1.put_data(np.array([[1, 2], [3, 4]]))
    buf2.put_data(np.array([[5, 6], [7, 8]]), start=2)

    print buf1
    print buf2

    try:
        dat = buf2.get_data(0, 5)
        dat[1, 4] = 9
    except ValueError:
        print("Write protected view on array")

    dat = buf2.get_data(0, 5, wprotect=False)
    dat[1, 4] = 9

    print buf1
    print buf2
