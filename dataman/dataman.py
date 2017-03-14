#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import sys
import cmd
import logging
import argparse

from .lib.constants import LOG_LEVEL_VERBOSE

__version__ = 0.1

NO_EXIT_CONFIRMATION = True
LOG_LEVEL = logging.INFO
log_level = LOG_LEVEL

class DataMan(cmd.Cmd):
    """Command line tool for quick data documentation."""

    prompt = "dm> "
    intro = "Data Manager\n"

    log = logging.getLogger(__name__)

    def preloop(self):
        self.log.debug("Starting DataMan CLI")

    def do_ls(self, line):
        parser = argparse.ArgumentParser('Recording statistics', prefix_chars='+/')
        parser.add_argument('path', help='Relative or absolute path to directory',
                default='.', nargs='?')
        cli_args = parser.parse_args(line.split(' ') if line else '')
        path = cli_args.path

        import dataman.lib.dirstats as ds
        ds.print_table(ds.gather(path))

    def do_stats(self, line):
        parser = argparse.ArgumentParser('Recording statistics', prefix_chars='+/')
        parser.add_argument('path', help='Relative or absolute path to directory',
                default='.', nargs='?')
        cli_args = parser.parse_args(line.split(' ') if line else '')
        path = cli_args.path

        import dataman.lib.dirstats as ds
        ds.print_table(ds.gather(path))

    def do_vis(self, line):
        from dataman.vis import vis
        vis.run(line.split(' '))

    def do_convert(self, line):
        pass

    def do_proc(self, line):
        print(sys.argv)

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
    parser = argparse.ArgumentParser(prog="DataMan")
    parser.add_argument('-d', '--debug', action='store_true',
            help='Debug mode -- verbose output, no confirmations.')
    parser.add_argument('--version', action='version', version='%(prog)s {version}'.format(version=__version__))

    # sub-parsers
    subparsers = parser.add_subparsers(help='sub commands', dest='command')

    parser_cli = subparsers.add_parser('cli', help='Interactive CLI session')
    parser_stats = subparsers.add_parser('stats', help='Recording stats (number channels, duration, sampling rate...')
    parser_ls = subparsers.add_parser('ls', help='Directory listing with basic information (e.g. size)')
    parser_vis = subparsers.add_parser('vis', help='Launch simple visualizer on data')
    parser_proc = subparsers.add_parser('proc', help='(Pre-)processing [NI}')
    parser_doc = subparsers.add_parser('doc', help='Documentation for prosperity [NI}')
    parser_check = subparsers.add_parser('check', help='Check/verify data and documentation integrity [NI}')
    parser_convert = subparsers.add_parser('convert', help='Convert into a different file format. [NI]')

    cli_args, cmd_args = parser.parse_known_args()

    if cli_args.debug:
        NO_EXIT_CONFIRMATION = True

    logging.addLevelName(LOG_LEVEL_VERBOSE, 'VERBOSE')
    log_level = LOG_LEVEL_VERBOSE if cli_args.debug else LOG_LEVEL
    logging.basicConfig(level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log = logging.getLogger(__name__)

    log.debug('CLI_ARGS: {}'.format(cli_args))
    log.debug('CMD_ARGS: {}'.format(cmd_args))

    # start cli
    if cli_args.command in [None, 'cli']:
        try:
            dm = DataMan().cmdloop()
        except KeyboardInterrupt:
            pass

    # some other command was given
    else:
        print('{} {:}'.format(cli_args.command, ' '.join(cmd_args)))
        DataMan().onecmd('{} {}'.format(cli_args.command, ' '.join(cmd_args)))


if __name__ == "__main__":
    main()

