#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import sys
import cmd
import logging

from lib.constants import LOG_LEVEL_VERBOSE
import lib.tools
from dataman_cli import DataMan

__version__ = 0.01

NO_EXIT_CONFIRMATION = True
LOG_LEVEL = logging.INFO
if __name__ == "__main__":
    # Command line parsing
    import argparse
    parser = argparse.ArgumentParser(prog="DataMan")
    parser.add_argument('-d', '--debug', action='store_true',
            help='Debug mode -- verbose output, no confirmations.')
    parser.add_argument('--version', action='version', version='%(prog)s {version}'.format(version=__version__))

    # sub-parsers
    subparsers = parser.add_subparsers(help='sub commands', dest='command')

    parser_cli = subparsers.add_parser('cli', help='Interactive CLI session')

    parser_stats = subparsers.add_parser('stats', help='Dataset statistics.')
    parser_stats.add_argument('path', help='Relative or absolute path to directory',
            default='.', nargs='?')

    parser.ls = subparsers.add_parser('ls', help='Directory listing with basic stats (e.g. size)')
    parser.ls .add_argument('path', help='Relative or absolute path to directory',
            default='.', nargs='?')

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
        DataMan().cmdloop()
    else:
        DataMan().onecmd(' '.join(sys.argv[1:]))

