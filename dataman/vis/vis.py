#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Multiple real-time digital signals with GLSL-based clipping.
"""

from __future__ import division

import logging
import os
import time
from multiprocessing import Queue

import numpy as np
import math

from vispy import gloo
from vispy import app
from vispy.util import keys

from Buffer import Buffer
from Streamer import Streamer

from reader import read_header

# Load vertex and fragment shaders
SHADER_PATH = os.path.join(os.path.dirname(__file__), 'shaders')
with open(os.path.join(SHADER_PATH, 'vis.vert')) as vs:
    VERT_SHADER = vs.read()
with open(os.path.join(SHADER_PATH, 'vis.frag')) as fs:
    FRAG_SHADER = fs.read()


class Vis(app.Canvas):
    """Simple visualizer for electrophysiology data.
        Parameters
    ----------
    buffer_size : int, optional
        buffer capacity (in samples)
    """
    def __init__(self, target, buffer_size=300000, proc_node=100):
        app.Canvas.__init__(self, title='dataman Vis',
                            keys='interactive')
        self.logger = logging.getLogger("Vis")
        self.running = False

        # Data source
        # TODO: Detect format (.dat, .continuous, .kwik), see dataman tools
        # TODO: Get number of channels
        # TODO: Get proc node of recording processor
        self.__target = target
        self.__target_proc_node = proc_node
        self.__target_hdr = read_header(os.path.join(self.target, '{}_CH1.continuous'.format(proc_node)))
        self.__n_samples_total = self.__target_hdr['numSamples']
        self.__n_channels = 64
        self.target_info()

        self.offset = 0
        self.drag_offset = 0

        # Dimensions of plot segment/signals
        self.n_cols = 1
        self.n_rows = int(self.n_channels/self.n_cols)
        # FIXME: n_samples should in the end refer to n_samples_total and the current depend on buffer size ans u_scale
        self.n_samples = 30000

        # Buffer to store all the pre-loaded signals
        self.__buffer_size = buffer_size
        self.__buf = Buffer()
        self.__buf.initialize(n_channels=self.n_channels, buf_size_samples=self.n_samples, np_type='float32')
        # self.__buf.put_data(np.zeros((self.n_channels, self.n_samples), dtype=np.float32))

        # Streamer to keep buffer filled
        self.__streamer = None
        self.q = Queue()
        self.start_streaming()

        # Color of each vertex
        # TODO: make it more efficient by using a GLSL-based color map and the index.
        self.color = np.repeat(np.random.uniform(size=(self.n_rows, 3), low=.1, high=.9),
                               self.n_samples*self.n_cols, axis=0).astype(np.float32)

        # Signal 2D index of each vertex (row and col) and x-index (sample index
        # within each signal).
        self.index = np.c_[np.repeat(np.repeat(np.arange(self.n_cols), self.n_rows), self.n_samples),
                           np.repeat(np.tile(np.arange(self.n_rows), self.n_cols), self.n_samples),
                           np.tile(np.arange(self.n_samples), self.n_channels)].astype(np.float32)

        self.program = gloo.Program(VERT_SHADER, FRAG_SHADER)
        self.program['a_position'] = self.__buf.get_data(0, self.n_samples).reshape(-1, 1)
        self.program['a_color'] = self.color
        self.program['a_index'] = self.index
        self.program['u_scale'] = (1., 1.)
        self.program['u_size'] = (self.n_rows, self.n_cols)
        self.program['u_n'] = self.n_samples

        gloo.set_viewport(0, 0, *self.physical_size)

        self._timer = app.Timer('auto', connect=self.on_timer, start=True)

        gloo.set_state(clear_color='black', blend=True,
                       blend_func=('src_alpha', 'one_minus_src_alpha'))

        self.show()

    def __get_is_streaming(self):
        try:
            return self.__streamer.is_alive()
        except:
            return False

    target = property(lambda self: self.__target, None, None,
                      "Target directory/file to stream from, read-only (string)")
    target_hdr = property(lambda self: self.__target_hdr, None, None,
                          "Header read from one of the data files containing meta data, read-only (dict)")
    n_channels = property(lambda self: self.__n_channels, None, None,
                          "Number of channels in the data set, read-only (int)")
    is_streaming = property(__get_is_streaming, None, None,
                            'Checks whether the data source streamer is active, read-only (bool)')
    buffer_size = property(lambda self: self.__buffer_size, None, None,
                           'Buffer capacity in samples, read-only (int)')

    def start_streaming(self):
        """Start streaming data into the shared buffer.
        """
        self.logger.info("Spawning streaming process...")
        self.__streamer = Streamer(target=self.target, queue=self.q,
                                   raw=self.__buf.raw, proc_node=self.__target_proc_node)
        self.__streamer._daemonic = True
        self.__streamer.start()
        self.logger.info("Streamer started")

    def stop_streaming(self):
        """Stop streaming by sending stop signal to the Streamer process
        """
        if not self.is_streaming:
            raise Exception("Streamer already stopped.")

        self.q.put(('stop', ))
        time.sleep(0.1)
        self.__streamer.join()
        self.logger.info("Streamer stopped")

    def target_info(self):
        """Check if target exists (and the format)"""
        fs = self.target_hdr['sampleRate']
        n_blocks = self.__n_samples_total/1024
        self.logger.info('Fs = {}kHz, {:.2f} records, {:.0f} samples, {:02.0f}min:{:02.0f}s'
                         .format(fs/1e3, n_blocks, self.__n_samples_total,
                                 math.floor(self.__n_samples_total/fs / 60),
                                 math.floor(self.__n_samples_total/fs % 60)))

    def set_scale(self, factor_x=1.0, factor_y=1.0, scale_x=None, scale_y=None):
        """Change visualization scaling of the subplots.
        Scaling is performed by the vertex shader.
        """
        scale_x_old, scale_y_old = self.program['u_scale']
        scale_x = scale_x_old if scale_x is None else scale_x
        scale_y = scale_y_old if scale_y is None else scale_y
        scale_x_new, scale_y_new = (scale_x * factor_x,
                                    scale_y * factor_y)
        self.program['u_scale'] = (max(1, scale_x_new), max(.05, scale_y_new))

    def set_offset(self, relative=0, absolute=0):
        """Crappy helper while seeking is still done in records instead of samples.
        """
        self.offset = absolute or self.offset
        self.offset += relative
        if self.offset < 0:
            self.offset = 0
        elif self.offset >= self.__n_samples_total/1024:
            self.offset = self.__n_samples_total/1024-1

    @staticmethod
    def on_resize(event):
        """Adjust viewport when window is resized. Smoothly does everything
        needed to adjust the graphs. Awesome.
        """
        gloo.set_viewport(0, 0, *event.physical_size)

    def on_key_press(self, event):
        """Keyboard handling."""
        # print event.key
        if event.key == keys.SPACE:
            self.running = not self.running
        elif event.key == 'Q':
            self.close()
        elif event.key == keys.LEFT:
            self.offset -= 2
            self.running = False
        elif event.key == keys.RIGHT:
            self.offset += 2
            self.running = False

    def on_mouse_move(self, event):
        """Handle mouse drag and hover"""
        if event.is_dragging:
            trail = event.trail()
            width = self.size[0]/self.n_cols
            height = self.size[1]/self.n_rows
            dx = trail[-1][0]-trail[0][0]
            dy = trail[-1][1]-trail[0][1]

            # drag signals
            if event.button == 1:
                shift_signal = dx/width
                shift_samples = shift_signal * self.n_samples
                shift_offset = int(shift_samples/1024)
                self.set_offset(absolute=self.drag_offset-shift_offset)
                self.running = False

            # change scaling
            if event.button == 2:
                self.set_scale(scale_x=1.0*math.exp(dx/width),
                               scale_y=1.0*math.exp(dy/height))

    def on_mouse_press(self, _):
        self.drag_offset = self.offset

    def on_mouse_wheel(self, event):
        """Mouse wheel control.

        Mouse wheel moves signal along x-axis.
        Shift+MW: x-axis scale (time scale),
        Ctrl+MW:  y-axis scale (amplitude)
        """
        if not len(event.modifiers):
            dx = -np.sign(event.delta[1])*int(event.delta[1]**2)
            self.set_offset(relative=dx*5)
            self.running = False
        else:
            delta = np.sign(event.delta[1]) * .05
            if keys.SHIFT in event.modifiers:
                self.set_scale(factor_x=math.exp(2.5*delta))
            elif keys.CONTROL in event.modifiers:
                self.set_scale(factor_y=math.exp(2.5*delta))

        self.update()

    def on_timer(self, _):
        """Update offset, move data from source buffer into GPU"""
        if self.is_streaming:
            self.q.put(('position', self.offset))

        # TODO: Only update the GPU buffer if necessary, have shader shift vertices, too
        self.program['a_position'].set_data(self.__buf.get_data(0, self.n_samples))

        # TODO: Shift by delta-frame time * Fs
        if self.running:
            self.set_offset(relative=1)

        self.update()

    def on_draw(self, _):
        gloo.clear()
        self.program.draw('line_strip')
        #print(self.offset)

    def on_close(self, _):
        self.stop_streaming()


def run(*args, **kwargs):
    Vis(*args, **kwargs)
    app.run()

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                        #format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                        format='%(asctime)s [%(process)-5d:%(threadName)-10s] %(name)s: %(levelname)s: %(message)s')
    run(target='../../data/2014-10-30_16-07-29', proc_node=106)
