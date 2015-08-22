#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import logging
import tools
import cmd
from constants import LOG_LEVEL_VERBOSE

class DataMan(cmd.Cmd):
    """Command line tool for quick data documentation."""

    prompt = "dm> "
    intro = "Data Manager\n --Ronny's way of avoiding having to stare at spreadsheets."

    def preloop(self):
        self.log = logging.getLogger(__name__)
        self.log.debug("starting DataMan CLI")
        # process command line arguments etc.

    def do_greet(self, user):
        """greet [user name]
        Simple user greeting. When used in combination with a parameter, will
        respond with personalized greeting. Yay."""
        if user:
            print("hello ", user)
        else:
            print("hi there!")

    def do_stats(self, path):
        if not len(path):
            path = '.'
        import folderstats as fs
        fs.print_table(fs.gather(path))

    def do_exit(self, line):
        "Exit"
        return True

    def do_EOF(self, line):
        "Exit"
        return True

    def postloop(self):
        print("Done.")

if __name__ == "__main__":
    logging.addLevelName(LOG_LEVEL_VERBOSE, "VERBOSE")
    logging.basicConfig(level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log = logging.getLogger(__name__)

    if len(sys.argv) > 1:
        DataMan().onecmd(' '.join(sys.argv[1:]))
    else:
        DataMan().cmdloop()
