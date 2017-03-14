#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vispy: gallery 2
# Copyright (c) 2015, Vispy Development Team.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.

"""
Multiple real-time digital signals with GLSL-based clipping.
"""

import logging
import os
import sys
from vispy import gloo
from vispy import app
from vispy import util
import numpy as np
import math

from ..lib.open_ephys import read_record
from ..lib import open_ephys, tools

from oio import util as oio_util

# Load vertex and fragment shaders
SHADER_PATH = os.path.join(os.path.dirname(__file__), 'shaders')
with open(os.path.join(SHADER_PATH, 'vis.vert')) as vs:
    VERT_SHADER = vs.read()
with open(os.path.join(SHADER_PATH, 'vis.frag')) as fs:
    FRAG_SHADER = fs.read()


class Vis(app.Canvas):
    def __init__(self, target_dir, n_cols=1, n_channels=64, max_samples_visible=30000, channels=None, bad_channels=None):
        app.Canvas.__init__(self, title='Use your wheel to zoom!', keys='interactive', size=(1920, 1080),
                            position=(0, 0), app='pyqt5')
        self.logger = logging.getLogger("Vis")
        # running traces, looks cool, but useless for the most part
        self.running = False
        self.offset = 0
        self.drag_offset = 0

        # Target configuration (format, sampling rate, sizes...)
        self.cfg = self._get_target_config(target_dir)
        self.target_dir = target_dir
        self.fs = self.cfg['HEADER']['sampling_rate']
        self.node_id = self.cfg['FPGA_NODE']
        self.n_blocks = int(self.cfg['HEADER']['n_blocks'])
        self.block_size = int(self.cfg['HEADER']['block_size'])
        self.n_samples_total = int(self.cfg['HEADER']['n_samples'])
        self.max_samples_visible = int(max_samples_visible)
        self.duration_total = tools.fmt_time(self.n_samples_total / self.fs)

        self.logger.debug(self.cfg)

        # FIXME: Get maximum number of channels from list of valid file names
        if not (n_channels) or n_channels < 1:
            self.n_channels = int(64)
        else:
            self.n_channels = int(n_channels)

        self.channel_order = channels  # if None: no particular order

        self.n_cols = int(n_cols)
        self.n_rows = int(math.ceil(self.n_channels / self.n_cols))

        self.logger.info('n_channels: {}, col/row: {}, buffer_size: {}, '
                         ' total_samples: {}, total_duration: {}'.format(self.n_channels,
                                                                         (self.n_cols, self.n_rows),
                                                                         self.max_samples_visible,
                                                                         self.n_samples_total,
                                                                         self.duration_total))

        # Buffer to store all the pre-loaded signals
        self.buf = np.zeros((self.n_channels, self.max_samples_visible), dtype=np.float32)

        # Color of each vertex
        # TODO: make it more efficient by using a GLSL-based color map and the index.
        color = np.repeat(np.random.uniform(size=(self.n_rows // 4, 3),
                                            low=.1, high=.9),
                          self.max_samples_visible * self.n_cols * 4, axis=0).astype(np.float32)

        # Load a nice color map instead of the random colors
        # cmap_path = os.path.join(os.path.join(os.path.dirname(__file__), 'shaders'), '4x4x8_half_vega20c_cmap.csv')
        # cmap = np.loadtxt(cmap_path, delimiter=',')
        # colors = np.repeat(cmap[:self.n_channels])
        # print(color.shape, cmap.shape)

        # Signal 2D index of each vertex (row and col) and x-index (sample index
        # within each signal).
        # FIXME: Build from lists for readability
        index = np.c_[np.repeat(np.repeat(np.arange(self.n_cols), self.n_rows), self.max_samples_visible),
                      np.repeat(np.tile(np.arange(self.n_rows), self.n_cols), self.max_samples_visible),
                      np.tile(np.arange(self.max_samples_visible), self.n_channels)] \
            .astype(np.float32)

        # Most of the magic happens in the vertex shader, moving the samples into "position" using
        # an affine transform based on number of columns and rows for the plot, scaling, etc.
        self.program = gloo.Program(VERT_SHADER, FRAG_SHADER)
        # FIXME: Reshaping not necessary?
        self.program['a_position'] = self.buf  # .reshape(-1, 1)
        self.program['a_color'] = color
        self.program['a_index'] = index
        self.program['u_scale'] = (1., 1.)
        self.program['u_size'] = (self.n_rows, self.n_cols)
        self.program['u_n'] = self.max_samples_visible

        gloo.set_viewport(0, 0, *self.physical_size)

        # sys.exit(0)
        self._timer = app.Timer('auto', connect=self.on_timer, start=True)

        gloo.set_state(clear_color='black', blend=True,
                       blend_func=('src_alpha', 'one_minus_src_alpha'))

        self.show()

    def _get_target_config(self, target_dir):
        # Check if we actually have data in there...
        data_format = open_ephys.detect(target_dir)

        if not (data_format) or not ('OE_' in data_format):
            self.logger.error('No valid open ephys .continuous data found at {}'.format(target_dir))
            sys.exit(1)
        self.logger.debug('Target found: {}'.format(data_format))

        return open_ephys.config(target_dir)

    def set_scale(self, factor_x=1.0, factor_y=1.0, scale_x=None, scale_y=None):
        scale_x_old, scale_y_old = self.program['u_scale']
        scale_x = scale_x_old if scale_x is None else scale_x
        scale_y = scale_y_old if scale_y is None else scale_y
        scale_x_new, scale_y_new = (scale_x * factor_x,
                                    scale_y * factor_y)
        self.program['u_scale'] = (max(1, scale_x_new), max(.05, scale_y_new))

    def set_offset(self, relative=0, absolute=0):
        """ Offset in blocks of 1024 samples """
        self.offset = int(absolute or self.offset)
        self.offset += int(relative)
        if self.offset < 0:
            self.offset = 0
        elif self.offset >= (self.n_samples_total - self.max_samples_visible) // self.block_size:
            self.offset = (self.n_samples_total - self.max_samples_visible) // self.block_size
        self.logger.debug('Block offset: {}, @ {}'.format(self.offset, tools.fmt_time(self.offset*1024/self.fs)))

    def on_resize(self, event):
        gloo.set_viewport(0, 0, *event.physical_size)

    def on_key_press(self, event):
        if event.key == 'Space':
            self.running = not self.running
        elif event.key == 'Q':
            self.close()
        elif event.key in ['Left', 'Right']:
            delta = 2

            # Increase jump width
            # TODO: Jump screen-multiples?
            if util.keys.SHIFT in event.modifiers:
                delta = delta * 10

            # Jump to beginning
            if util.keys.CONTROL in event.modifiers:
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
                shift_samples = shift_signal * self.max_samples_visible
                shift_offset = int(shift_samples / 1024)
                self.set_offset(absolute=self.drag_offset - shift_offset)

            if event.button == 2:
                self.set_scale(scale_x=1.0 * math.exp(dx / width),
                               scale_y=1.0 * math.exp(dy / height))

    def on_mouse_press(self, event):
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
            if util.keys.SHIFT in event.modifiers:
                self.set_scale(factor_x=math.exp(2.5 * delta))

            elif util.keys.CONTROL in event.modifiers:
                self.set_scale(factor_y=math.exp(2.5 * delta))

        self.update()

    def on_mouse_double_click(self, event):
        x, y = event.pos
        x_r = x / self.size[0]
        y_r = y / self.size[1]

        # TODO: use y-coordinate to guesstimate the channel id + amplitude at point
        t_r = x_r * self.n_cols - math.floor(x_r * self.n_cols)
        t_sample = (t_r * self.max_samples_visible + self.offset * self.block_size)
        t_sec = t_sample / self.fs
        self.logger.info('Sample {} @ {}'.format(int(t_sample), tools.fmt_time(t_sec)))

    def on_timer(self, event):
        """Frame update callback."""
        # FIXME: Sample precision positions
        # FIXME: Only read in data when needed, not per frame. Duh. :D

        # If nothing got dirty, change nothing
        # If scale or offset changed, move, move move
        #   If still a healthy buffer, don't do anything
        #   If running out of buffer, append or prepend new data
        for i in range(self.n_channels):
            chan_id = i if self.channel_order is None else self.channel_order[i]
            self.buf[i, :self.max_samples_visible] = \
                read_record(os.path.join(self.target_dir, '{node_id}_CH{channel}.continuous'.format(
                    node_id=self.node_id,
                    channel=chan_id + 1)),
                            offset=self.offset)[:self.max_samples_visible]
        self.program['a_position'].set_data(self.buf)

        if self.running:
            self.set_offset(relative=1)

        self.update()

    def on_draw(self, event):
        gloo.clear()
        self.program.draw('line_strip')


def run(*args, **kwargs):
    import argparse
    parser = argparse.ArgumentParser('Data Visualization', prefix_chars='+')
    parser.add_argument('path', help='Relative or absolute path to directory',
                        default='.', nargs='?')
    parser.add_argument('+c', '++cols', help='Number of columns', default=1, type=int)
    parser.add_argument('+C', '++channels', help='Number of channels', default=64, type=int)
    parser.add_argument('+l', '++layout', help='Path to probe file defining channel order')

    cli_args = parser.parse_args(*args)
    if 'layout' in cli_args and cli_args.layout is not None:
        layout = oio_util.run_prb(cli_args.layout)
        channels, bad_channels = oio_util.flat_channel_list(layout)[:cli_args.channels]
    else:
        channels = None
        bad_channels = None

    vis = Vis(cli_args.path,
              n_cols=cli_args.cols,
              n_channels=cli_args.channels,
              channels=channels,
              bad_channels=bad_channels)
    app.run()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    run('../../data/2014-10-30_16-07-29')
