#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import sys
import cmd
import logging

from .lib.constants import LOG_LEVEL_VERBOSE

__version__ = 0.01

NO_EXIT_CONFIRMATION = True
LOG_LEVEL = logging.INFO


class DataMan(cmd.Cmd):
    """Command line tool for quick data documentation."""

    prompt = "dm> "
    intro = "Data Manager\n"

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

    def do_ls(self, path):
        if not len(path):
            path = '.'
        import dataman.lib.dirstats as ds
        ds.print_table(ds.gather(path))

    def do_stats(self, path):
        if not len(path):
            path = '.'
        import dataman.lib.dirstats as ds
        ds.print_table(ds.gather(path))

    def do_vis(self, path):
        from dataman.vis import vis
        vis.run(target=path)

    def do_exit(self, line):
        "Exit"
        return True

    def do_EOF(self, line):
        "Exit"
        return True

    def postloop(self):
        print("Done.")


def main():
    # Command line parsing
    import argparse
    parser = argparse.ArgumentParser(prog="DataMan")
    parser.add_argument('-d', '--debug', action='store_true',
            help='Debug mode -- verbose output, no confirmations.')
    parser.add_argument('--version', action='version', version='%(prog)s {version}'.format(version=__version__))

    # sub-parsers
    subparsers = parser.add_subparsers(help='sub commands', dest='command')

    # CLI
    parser_cli = subparsers.add_parser('cli', help='Interactive CLI session')

    # STATS
    parser_stats = subparsers.add_parser('stats', help='Dataset stats (number channels, duration, sampling rate...')
    parser_stats.add_argument('path', help='Relative or absolute path to directory',
            default='.', nargs='?')

    # LS
    parser_ls = subparsers.add_parser('ls', help='Directory listing with basic information (e.g. size)')
    parser_ls .add_argument('path', help='Relative or absolute path to directory',
            default='.', nargs='?')

    # VIS
    parser_vis = subparsers.add_parser('vis', help='Launch simple visualizer on dataset')
    parser_vis.add_argument('path', help='Relative or absolute path to directory',
                            default='.', nargs='?')

    # PROC, DOC, CHECK
    parser_proc = subparsers.add_parser('proc', help='Data processing')
    parser_doc = subparsers.add_parser('doc', help='Data documentation')
    parser_check = subparsers.add_parser('check', help='Check/verify data and documentation integrity')

    cli_args = None
    if len(sys.argv) > 1:
        cli_args = parser.parse_args()
        if cli_args.debug:
            NO_EXIT_CONFIRMATION = True

    log_level = LOG_LEVEL_VERBOSE if cli_args is not None and cli_args.debug else LOG_LEVEL
    logging.addLevelName(LOG_LEVEL_VERBOSE, "VERBOSE")
    logging.basicConfig(level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log = logging.getLogger(__name__)

    if cli_args is None or cli_args.command == 'cli':
        try:
            dm = DataMan().cmdloop()
        except KeyboardInterrupt:
            pass
    else:
        DataMan().onecmd(' '.join(sys.argv[1:]))


if __name__ == "__main__":
    main()

