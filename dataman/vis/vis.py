#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vispy: gallery 2
# Copyright (c) 2015, Vispy Development Team.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.

"""
Multiple real-time digital signals with GLSL-based clipping.
"""
from __future__ import division
import logging
import os
from vispy import gloo
from vispy import app
from vispy import util
import numpy as np
import math

from reader import read_record, read_header

# Number of cols and rows in the table.
nrows = 16
ncols = 4

# Number of signals.
m = nrows*ncols

# Number of samples per signal.
# FIXME: Depends on zoom level/sampling rate?
n = 3e4

# Buffer
buf = np.zeros((m, n), dtype=np.float32)

# Color of each vertex
# TODO: make it more efficient by using a GLSL-based color map and the index.
color = np.repeat(np.random.uniform(size=(nrows, 3), low=.1, high=.9),
                  n*ncols, axis=0).astype(np.float32)

# Signal 2D index of each vertex (row and col) and x-index (sample index
# within each signal).
index = np.c_[np.repeat(np.repeat(np.arange(ncols), nrows), n),
              np.repeat(np.tile(np.arange(nrows), ncols), n),
              np.tile(np.arange(n), m)].astype(np.float32)

with open('shaders/vis.vert') as vs:
    VERT_SHADER = vs.read()
with open('shaders/vis.frag') as fs:
    FRAG_SHADER = fs.read()


class Vis(app.Canvas):
    def __init__(self, target):
        app.Canvas.__init__(self, title='Use your wheel to zoom!',
                            keys='interactive')
        self.logger = logging.getLogger("Vis")
        self.running = False
        self.offset = 0
        self.drag_offset = 0

        self.target = target
        self.test_target()

        self.program = gloo.Program(VERT_SHADER, FRAG_SHADER)
        self.program['a_position'] = buf.reshape(-1, 1)
        self.program['a_color'] = color
        self.program['a_index'] = index
        self.program['u_scale'] = (1., 1.)
        self.program['u_size'] = (nrows, ncols)
        self.program['u_n'] = n

        gloo.set_viewport(0, 0, *self.physical_size)

        self._timer = app.Timer('auto', connect=self.on_timer, start=True)

        gloo.set_state(clear_color='black', blend=True,
                       blend_func=('src_alpha', 'one_minus_src_alpha'))

        self.show()

    def test_target(self):
        fname = os.path.join(self.target, '106_CH1.continuous')
        self.logger.info("Reading file header of {}".format(fname))
        hdr = read_header(fname)
        fs = hdr['sampleRate']
        n_blocks = (os.path.getsize(fname)-1024)/2070
        n_samples = n_blocks*1024
        self.logger.info('Fs = {}kHz, {} blocks, {:.0f} samples, {:02.0f}min:{:02.0f}s'
                         .format(fs/1e3, n_blocks, n_samples,
                                 math.floor(n_samples/fs/60),
                                 math.floor(n_samples/fs%60)))

    def set_scale(self, factor_x=1.0, factor_y=1.0, scale_x=None, scale_y=None):
        scale_x_old, scale_y_old = self.program['u_scale']
        scale_x = scale_x_old if scale_x is None else scale_x
        scale_y = scale_y_old if scale_y is None else scale_y
        scale_x_new, scale_y_new = (scale_x * factor_x,
                                    scale_y * factor_y)
        self.program['u_scale'] = (max(1, scale_x_new), max(.05, scale_y_new))

    def set_offset(self, relative=0, absolute=0):
        self.offset = absolute or self.offset
        self.offset += relative
        if self.offset < 0:
            self.offset = 0
        elif self.offset > 1000:
            self.offset = 1000

    def on_resize(self, event):
        gloo.set_viewport(0, 0, *event.physical_size)

    def on_key_press(self, event):
        # print event.key
        if event.key == 'Space':
            self.running = not self.running
        elif event.key == 'Q':
            self.close()
        elif event.key == 'Left':
            self.offset = max(0, self.offset-2)
        elif event.key == 'Right':
            self.offset = min(100, self.offset+2)

    def on_mouse_move(self, event):
        """Handle mouse drag and hover"""
        if event.is_dragging:
            trail = event.trail()
            width = self.size[0]/ncols
            height = self.size[1]/nrows
            dx = trail[-1][0]-trail[0][0]
            dy = trail[-1][1]-trail[0][1]

            if event.button == 1:
                shift_signal = dx/width
                shift_samples = shift_signal * n
                shift_offset = int(shift_samples/1024)
                self.set_offset(absolute=self.drag_offset-shift_offset)

            if event.button == 2:
                self.set_scale(scale_x=1.0*math.exp(dx/width),
                               scale_y=1.0*math.exp(dy/height))

    def on_mouse_press(self, event):
        self.drag_offset = self.offset

    def on_mouse_wheel(self, event):
        """Mouse wheel control.

        Mouse wheel moves signal along x-axis.
        Shift+MW: x-axis scale (time scale),
        Ctrl+MW:  y-axis scale (amplitude)
        """
        if not len(event.modifiers):
            dx = -np.sign(event.delta[1])*int(event.delta[1]**2)
            self.set_offset(relative=dx)
        else:
            delta = np.sign(event.delta[1]) * .05
            if util.keys.SHIFT in event.modifiers:
                self.set_scale(factor_x=math.exp(2.5*delta))
            elif util.keys.CONTROL in event.modifiers:
                self.set_scale(factor_y=math.exp(2.5*delta))

        self.update()

    def on_timer(self, event):
        """Add some data at the end of each signal (real-time signals)."""
        # FIXME: Sample precision positions
        # FIXME: Only read in data when needed, not per frame. Duh. :D
        for i in range(m):
            buf[i, :n] = read_record('data/2014-10-30_16-07-29/106_CH{}.continuous'.format(i+1),
                                     offset=self.offset)[:n]
        self.program['a_position'].set_data(buf)

        if self.running:
            self.set_offset(relative=1)

        self.update()

    def on_draw(self, event):
        gloo.clear()
        self.program.draw('line_strip')

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    c = Vis(target='data/2014-10-30_16-07-29')
    app.run()
