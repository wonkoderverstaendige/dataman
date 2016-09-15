#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import sys

from dataman.cli.cli import DataManCLI
from dataman.lib.constants import LOG_LEVEL_VERBOSE

__version__ = '0.02dev'

NO_EXIT_CONFIRMATION = True
DEFAULT_LOG_LEVEL = logging.INFO


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
    parser_stats = subparsers.add_parser('stats', help='Dataset statistics.')
    parser_stats.add_argument('path', help='Relative or absolute path to directory',
                              default='.', nargs='?')

    # LS
    parser_ls = subparsers.add_parser('ls', help='Directory listing with basic stats (e.g. size)')
    parser_ls.add_argument('path', help='Relative or absolute path to directory',
                           default='.', nargs='?')

    # VIS
    parser_vis = subparsers.add_parser('vis', help='Launch simple visualizer on dataset')
    parser_vis.add_argument('path', help='Relative or absolute path to directory',
                            default='.', nargs='?')

    # PROC, DOC, CHECK
    parser_proc = subparsers.add_parser('proc', help='Data processing')
    parser_doc = subparsers.add_parser('doc', help='Data documentation')
    parser_check = subparsers.add_parser('check', help='Check/verify data and documentation integrity')

    cli_args = parser.parse_args()

    log_level = LOG_LEVEL_VERBOSE if cli_args is not None and cli_args.debug else DEFAULT_LOG_LEVEL
    logging.addLevelName(LOG_LEVEL_VERBOSE, "VERBOSE")
    logging.basicConfig(level=log_level,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log = logging.getLogger(__name__)

    if cli_args.command in [None, 'cli']:
        try:
            DataManCLI().cmdloop()
        except KeyboardInterrupt:
            pass
    else:
        DataManCLI().onecmd(' '.join(sys.argv[1:]))


if __name__ == "__main__":
    main()

