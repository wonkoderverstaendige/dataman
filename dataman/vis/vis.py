#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vispy: gallery 2
# Copyright (c) 2015, Vispy Development Team.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.

"""
Multiple real-time digital signals with GLSL-based clipping.
"""

import logging
import math
import os
import os.path as op
import sys
import time
from multiprocessing import Queue

import numpy as np
from oio import util as oio_util
from vispy import app
from vispy import gloo
from vispy.util import keys

from dataman.lib.SharedBuffer import SharedBuffer
from ..lib import open_ephys, dat, kwik, tools

# Load vertex and fragment shaders
SHADER_PATH = os.path.join(os.path.dirname(__file__), 'shaders')
with open(os.path.join(SHADER_PATH, 'vis.vert')) as vs:
    VERT_SHADER = vs.read()
with open(os.path.join(SHADER_PATH, 'vis.frag')) as fs:
    FRAG_SHADER = fs.read()

BUFFER_DTYPE = 'float32'
BUFFER_LENGTH = int(3e4)


class Vis(app.Canvas):
    def __init__(self, target_path, n_cols=1, channels=None, *args, **kwargs):
        app.Canvas.__init__(self, title='Use your wheel to zoom!', keys='interactive', size=(1920, 1080),
                            position=(0, 0), app='pyqt5')
        self.logger = logging.getLogger(__name__)

        # Target configuration (format, sampling rate, sizes...)
        self.target_path = target_path
        self.logger.debug('Target path: {}'.format(target_path))
        self.format = self._target_format()
        self.logger.debug('Target module: {}'.format(self.format))

        self.cfg = self._get_target_config(*args, **kwargs)
        self.logger.debug(self.cfg)

        self.fs = self.cfg['HEADER']['sampling_rate']
        self.n_channels = self.cfg['CHANNELS']['n_channels']
        self.channel_order = channels  # if None: no particular order

        self.n_samples_total = int(self.cfg['HEADER']['n_samples'])
        self.duration_total = tools.fmt_time(self.n_samples_total / self.fs)

        # Buffer to store all the pre-loaded signals
        self.buf = SharedBuffer()
        self.buffer_length = BUFFER_LENGTH
        self.buf.initialize(n_channels=self.n_channels, n_samples=self.buffer_length, np_dtype=BUFFER_DTYPE)

        # Streamer to keep buffer filled
        self.streamer = None
        self.stream_queue = Queue()
        self.start_streaming()

        # Setup up viewport and viewing state variables
        # running traces, looks cool, but useless for the most part
        self.running = False
        self.dirty = True
        self.offset = 0
        self.drag_offset = 0
        self.n_cols = int(n_cols)
        self.n_rows = int(math.ceil(self.n_channels / self.n_cols))

        self.logger.debug('n_channels: {}, col/row: {}, buffer_length: {}, '
                          ' total_samples: {}, total_duration: {}'.format(self.n_channels,
                                                                          (self.n_cols, self.n_rows),
                                                                          self.buffer_length,
                                                                          self.n_samples_total,
                                                                          self.duration_total))

        # Most of the magic happens in the vertex shader, moving the samples into "position" using
        # an affine transform based on number of columns and rows for the plot, scaling, etc.
        self.program = gloo.Program(VERT_SHADER, FRAG_SHADER)
        self._feed_shaders()

        gloo.set_viewport(0, 0, *self.physical_size)

        self._timer = app.Timer('auto', connect=self.on_timer, start=True)

        gloo.set_state(clear_color='black', blend=True,
                       blend_func=('src_alpha', 'one_minus_src_alpha'))
        self.show()

    def __get_is_streaming(self):
        # try:
        return self.streamer.is_alive()
        # except:
        #     return False

    # target = property(lambda self: self.__target, None, None,
    #                   "Target directory/file to stream from, read-only (string)")
    # target_hdr = property(lambda self: self.__target_hdr, None, None,
    #                       "Header read from one of the data files containing meta data, read-only (dict)")
    # n_channels = property(lambda self: self.__n_channels, None, None,
    #                       "Number of channels in the data set, read-only (int)")
    is_streaming = property(__get_is_streaming, None, None,
                            'Checks whether the data source streamer is active, read-only (bool)')

    def start_streaming(self):
        """Start streaming data into the shared buffer.
        """
        self.logger.debug("Spawning streaming process...")
        self.streamer = self.format.DataStreamer(queue=self.stream_queue, raw=self.buf.raw,
                                                 target_path=self.target_path)
        self.streamer._daemonic = True
        self.stream_queue.put(('offset', 0))
        self.streamer.start()

    def stop_streaming(self):
        """Stop streaming by sending stop signal to the Streamer process
        """
        if not self.is_streaming:
            self.logger.warning("Can't stop Streamer process: Already stopped.")
        else:
            self.stream_queue.put(('stop', None))
            time.sleep(0.1)
            self.streamer.join()
            self.logger.debug("Streamer stopped")

    def _feed_shaders(self):
        # Color of each vertex
        # TODO: make it more efficient by using a GLSL-based color map and the index.
        # Load a nice color map instead of the random colors
        # cmap_path = os.path.join(os.path.join(os.path.dirname(__file__), 'shaders'), '4x4x8_half_vega20c_cmap.csv')
        # cmap = np.loadtxt(cmap_path, delimiter=',')
        # colors = np.repeat(cmap[:self.n_channels])
        # print(color.shape, cmap.shape)
        group = min(self.n_channels, 4)
        color = np.repeat(np.random.uniform(size=(math.ceil(self.n_rows / 4), 3),
                                            low=.1, high=.9),
                          self.buffer_length * self.n_cols * group, axis=0).astype('float32')

        # Signal 2D index of each vertex (row and col) and x-index (sample index
        # within each signal).
        # index = np.c_[np.repeat(np.repeat(np.arange(self.n_cols), self.n_rows), self.max_samples_visible),
        #               np.repeat(np.tile(np.arange(self.n_rows), self.n_cols), self.max_samples_visible),
        #               np.tile(np.arange(self.max_samples_visible), self.n_channels)] \
        #     .astype(np.float32)

        def lr(n):
            return list(range(n))

        def flatten(l):
            return [item for sl in l for item in sl]

        col_idc = flatten([[r] * self.n_rows * self.buffer_length for r in lr(self.n_cols)])
        row_idc = flatten([[r] * self.buffer_length for r in lr(self.n_rows)]) * self.n_cols
        ch_idc = lr(self.buffer_length) * self.n_channels
        # needs a copy to make memory contiguous
        idc = np.transpose(np.array([col_idc, row_idc, ch_idc], dtype='float32'))

        self.program['a_position'] = np.ones((self.n_channels, self.buffer_length), dtype=BUFFER_DTYPE)
        self.program['a_color'] = color
        self.program['a_index'] = idc.copy()
        self.program['u_scale'] = (1., max(.1, 1. - 1 / self.n_channels))
        self.program['u_size'] = (self.n_rows, self.n_cols)
        self.program['u_n'] = self.buffer_length

    def _target_format(self):
        formats = [f for f in [fmt.detect(self.target_path) for fmt in [open_ephys, dat, kwik]] if f is not None]

        if len(formats) == 1:
            fmt = formats[0]
            if 'DAT' in fmt:
                if fmt == 'DAT-File':
                    return dat
            else:
                if 'kwik' in fmt:
                    return kwik
                else:
                    return open_ephys
        self.logger.info('Detected format(s) {} not valid.'.format(formats))
        sys.exit(0)

    def _get_target_config(self, *args, **kwargs):
        self.logger.debug('Target found: {}'.format(self.format.FMT_NAME))
        return self.format.config(self.target_path, *args, **kwargs)

    def set_scale(self, factor_x=1.0, factor_y=1.0, scale_x=None, scale_y=None):
        self.dirty = True
        scale_x_old, scale_y_old = self.program['u_scale']
        scale_x = scale_x_old if scale_x is None else scale_x
        scale_y = scale_y_old if scale_y is None else scale_y
        scale_x_new, scale_y_new = (scale_x * factor_x,
                                    scale_y * factor_y)
        u_scale = (max(1, scale_x_new), max(.05, scale_y_new))
        self.logger.debug('u_scale:{}'.format(u_scale))
        self.program['u_scale'] = u_scale

    def set_offset(self, relative=0, absolute=0):
        """ Offset in blocks of 1024 samples """
        old_offset = self.offset
        self.offset = int(absolute or self.offset)
        self.offset += int(relative)
        if self.offset < 0:
            self.offset = 0
        elif self.offset >= (self.n_samples_total - self.buffer_length) // self.cfg['HEADER']['block_size']:
            self.offset = (self.n_samples_total - self.buffer_length) // self.cfg['HEADER']['block_size']

        if old_offset != self.offset:
            self.dirty = True

    @staticmethod
    def on_resize(event):
        """Adjust viewport when window is resized. Smoothly does everything
        needed to adjust the graphs. Awesome.
        """
        gloo.set_viewport(0, 0, *event.physical_size)

    def on_key_press(self, event):
        if event.key == keys.SPACE:
            self.running = not self.running
        elif event.key == 'Q':
            self.close()
        elif event.key in [keys.LEFT, keys.RIGHT]:
            delta = 2

            # Increase jump width
            # TODO: Jump screen-multiples?
            if keys.SHIFT in event.modifiers:
                delta *= 10

            # Jump to beginning
            if keys.CONTROL in event.modifiers:
                delta = self.n_samples_total

            if event.key == 'Left':
                self.set_offset(relative=-delta)
            elif event.key == 'Right':
                self.set_offset(relative=delta)

    def on_mouse_move(self, event):
        """Handle mouse drag and hover"""
        if event.is_dragging:
            trail = event.trail()
            width = self.size[0] / self.n_cols
            height = self.size[1] / self.n_rows
            dx = trail[-1][0] - trail[0][0]
            dy = trail[-1][1] - trail[0][1]

            if event.button == 1:
                shift_signal = dx / width
                shift_samples = shift_signal * self.buffer_length
                shift_offset = int(shift_samples / 1024)
                self.set_offset(absolute=self.drag_offset - shift_offset)

            if event.button == 2:
                self.set_scale(scale_x=1.0 * math.exp(dx / width),
                               scale_y=1.0 * math.exp(dy / height))

    def on_mouse_press(self, _):
        self.drag_offset = self.offset

    def on_mouse_wheel(self, event):
        """Mouse wheel control.

        Mouse wheel moves signal along x-axis.
        Shift+MW: x-axis scale (time scale),
        Ctrl+MW:  y-axis scale (amplitude)
        """
        if not len(event.modifiers):
            dx = -np.sign(event.delta[1]) * int(event.delta[1] ** 2)
            self.set_offset(relative=dx)
        else:
            delta = np.sign(event.delta[1]) * .05
            if keys.SHIFT in event.modifiers:
                self.set_scale(factor_x=math.exp(2.5 * delta))

            elif keys.CONTROL in event.modifiers:
                self.set_scale(factor_y=math.exp(2.5 * delta))

        self.update()

    def on_mouse_double_click(self, event):
        x, y = event.pos
        x_r = x / self.size[0]
        # y_r = y / self.size[1]

        # TODO: use y-coordinate to guesstimate the channel id + amplitude at point
        t_r = x_r * self.n_cols - math.floor(x_r * self.n_cols)
        t_sample = (t_r * self.buffer_length + self.offset * self.cfg['HEADER']['block_size'])
        t_sec = t_sample / self.fs
        self.logger.info('Sample {} @ {}'.format(int(t_sample), tools.fmt_time(t_sec)))

    def on_timer(self, _):
        """Frame update callback."""
        if self.dirty:
            if self.is_streaming:
                self.stream_queue.put(('offset', self.offset))
            self.dirty = False

            data = self.buf.get_data(0, self.buffer_length)
            self.program['a_position'].set_data(data)

        if self.running:
            self.set_offset(relative=1)

        self.update()

    def on_draw(self, _):
        gloo.clear()
        self.program.draw('line_strip')

    def on_close(self, _):
        self.stop_streaming()


def run(*args, **kwargs):
    import argparse
    parser = argparse.ArgumentParser('Data Visualization',
                                     epilog="""Use: Mousewheel/arrow keys to scroll,
                                     <Shift>/<Ctrl>+<left>/<right> for larger jumps.
                                     <Shift>/<Ctrl>+Mousewheel to scale.
                                     Use <q> or <Esc> to exit. """)
    parser.add_argument('path', help='Relative or absolute path to directory',
                        default='.', nargs='?')
    parser.add_argument('-d', '--debug', action='store_true',
                        help='Debug mode -- verbose output, no confirmations.')
    parser.add_argument('-c', '--cols', help='Number of columns', default=1, type=int)
    parser.add_argument('-C', '--channels', help='Number of channels', type=int)
    parser.add_argument('-l', '--layout', help='Path to probe file defining channel order')
    parser.add_argument('-D', '--dtype', help='Data type if needed (e.g. float32 dat files')

    cli_args = parser.parse_args(*args)
    if 'layout' in cli_args and cli_args.layout is not None:
        layout = oio_util.run_prb(cli_args.layout)
        channels, bad_channels = oio_util.flat_channel_list(layout)[:cli_args.channels]
    else:
        channels = None
        bad_channels = None

    Vis(op.abspath(op.expanduser(cli_args.path)),
        n_cols=cli_args.cols,
        n_channels=cli_args.channels,
        channels=channels,
        bad_channels=bad_channels,
        dtype=cli_args.dtype)
    app.run()


if __name__ == "__main__":
    pass
