#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on Sep 24, 2015 15:32
@author: <'Ronny Eichler'> ronny.eichler@gmail.com

Stream data to buffer
"""

import os
import logging
import time
import signal
import datetime
from multiprocessing import Process
from reader import read_record, read_header

from Buffer import Buffer


# copy from lib.tools
def fmt_seconds(seconds):
    """Format seconds as a timestamp in HH:MM:SS.uuu format.
    Parameters:
        seconds : float
    """
    seconds, milliseconds = divmod(seconds, 1.)
    td_s = datetime.timedelta(seconds=seconds)
    td_ms = '{:.3f}'.format(milliseconds).lstrip('0')
    return '{:0>8}{}'.format(td_s, td_ms)


class Streamer(Process):
    """Process to stream data from file source.

    Controlled via commands submitted to a queue and read data
    is placed in a buffer shared with parent process.

    ----------
    buffer_size : int, optional
        buffer capacity (in samples)
    """
    def __init__(self, target, queue, raw, proc_node):
        self.logger = logging.getLogger(__name__)
        self.__buf = Buffer()
        self.__buf.initialize_from_raw(raw)
        self.q = queue  # Queue receives tuple of (command (string), data) or (command, None)
        self.position = None

        # TODO: target as class, with information about type, mapping, internal/external clocking
        self.target = target
        # TODO: Take channel map in the .prb file into account
        channel_list = xrange(self.__buf.n_channels)
        self.files = [(channel, os.path.join(self.target, '{}_CH{}.continuous'.format(proc_node, channel+1)))
                      for channel in channel_list]
        self.target_header = read_header(self.files[0][1])

        # dictionary of known commands
        self.cmds = {'stop': self.stop}

        super(Streamer, self).__init__()

    def run(self):
        """Main streaming loop."""
        cmd = self.__get_cmd()
        self.logger.info("Started streaming")

        # ignore CTRL+C, runs daemonic, will stop with parent
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        while cmd != 'stop':
            # Grab all messages currently in the queue
            messages = self.__get_cmd()
            pos_changes = [msg[1] for msg in messages if msg[0] == 'position' and msg[1] is not None]
            last_pos = pos_changes[-1] if len(pos_changes) else self.position
            cmd = 'stop' if 'stop' in [msg[0] for msg in messages if msg[0] != 'position'] else None

            if last_pos is not None and self.position != last_pos:
                self.position = last_pos

                # READ IN DATA
                # TODO: Here be worker pool of threads/processes grabbing data into the shared buffer
                # TODO: Avoid extra copy of data by having Buffer return view on array and write in place
                t = time.time()
                for sf in self.files:
                    data = read_record(sf[1], offset=self.position)[:self.__buf.buf_samples]
                    self.__buf.put_channel_data(data, channel=sf[0])
                self.logger.debug('Read {} channel data at position {} in {:.0f} ms'.
                                  format(self.__buf.n_channels,
                                         fmt_seconds(self.position*1024/self.target_header['sampleRate']),
                                         (time.time()-t)*1000))
            time.sleep(0.02)

        self.logger.info('Stopped streaming')

    def stop(self):
        pass

    def __get_cmd(self):
        messages = []
        for msg in range(0, self.q.qsize()):
            try:
                messages.append(self.q.get(False))
            except Exception as e:
                break
        return messages

    def __execute_cmd(self, cmd):
        if cmd in self.cmds:
            try:
                self.cmds[cmd]()
            except:
                self.logger.warning('unable to execute command {}'.format(cmd))


# ------------------------------------------------------------------------------
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='[%(process)-5d:%(threadName)-10s] %(name)s: %(levelname)s: %(message)s')
