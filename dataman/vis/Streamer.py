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
from multiprocessing import Process
from reader import read_record, read_header

from Buffer import Buffer


class Streamer(Process):
    def __init__(self, target, queue, raw):
        self.logger = logging.getLogger(__name__)
        self.__buf = Buffer()
        self.__buf.initialize_from_raw(raw)
        self.q = queue
        self.position = None

        self.target = target
        self.files = [os.path.join(self.target, '106_CH{}.continuous'.format(i+1)) for i in range(self.__buf.nChannels)]

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
            messages = self.__get_cmd()
            pos_changes = [msg[1] for msg in messages if msg[0] == 'position' and msg[1] is not None]
            last_pos = pos_changes[-1] if len(pos_changes) else self.position
            cmd = 'stop' if 'stop' in [msg[0] for msg in messages if msg[0] != 'position'] else None

            if last_pos is not None and self.position != last_pos:
                self.position = last_pos

                # READ IN DATA
                # TODO: Here be worker pool of threads/processes grabbing data into the shared buffer
                t = time.time()
                for i, f in enumerate(self.files):
                    data = read_record(f, offset=self.position)[:self.__buf.nSamples]
                    self.__buf.put_data(data, channel=i)
                self.logger.debug('Read data at position {} in {:.0f} ms'.format(self.position, (time.time()-t)*1000))
            time.sleep(0.02)

        self.logger.info('Stopped streaming')
        # cmd = self.__get_cmd()
        # self.__execute_cmd(cmd)

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
        # try:
        #     cmd, data = self.q.get(False)
        #     return cmd, data
        # except Exception:
        #     return None, None

    def __execute_cmd(self, cmd):
        if cmd in self.cmds:
            try:
                self.cmds[cmd]()
            except:
                self.logger.warning('unable to execute command {}'.format(cmd))


# ------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='[%(process)-5d:%(threadName)-10s] %(name)s: %(levelname)s: %(message)s')

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # client = Client(buffer_size=300000, buffer_window=10)
    # client.connect(('', 51244))
    # client.start_streaming()
    # time.sleep(10)
    # client.stop_streaming()
    # client.disconnect()