#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import cmd
import logging
import os.path as op
import sys

from .lib.constants import LOG_LEVEL_VERBOSE

__version__ = '0.1.1'

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
        parser = argparse.ArgumentParser('Recording statistics',)
        parser.add_argument('path', help='Relative or absolute path to directory',
                            default='.', nargs='?')
        parser.add_argument('-d', '--debug', action='store_true',
                            help='Debug mode -- verbose output, no confirmations.')

        cli_args = parser.parse_args(line.split(' ') if line else '')
        self.log.debug('ls with args: {}'.format(cli_args))
        path = op.abspath(op.expanduser(cli_args.path))
        self.log.debug('Expanded path: {}'.format(path))

        import dataman.dirstats.dirstats as ds
        ds.print_table(ds.gather(path))

    def do_stats(self, line):
        parser = argparse.ArgumentParser('Recording statistics')
        parser.add_argument('path', help='Relative or absolute path to directory',
                            default='.', nargs='?')
        parser.add_argument('-d', '--debug', action='store_true',
                            help='Debug mode -- verbose output, no confirmations.')
        cli_args = parser.parse_args(line.split(' ') if line else '')
        self.log.debug('Stats with args: {}'.format(cli_args))
        path = op.abspath(op.expanduser(cli_args.path))
        self.log.debug('Expanded path: {}'.format(path))

        import dataman.dirstats.dirstats as ds
        ds.print_table(ds.gather(path))

    def do_vis(self, line):
        from dataman.vis import vis
        vis.run(line.split(' '))

    def do_convert(self, line):
        pass

    def do_check(self, line):
        pass

    def do_proc(self, line):
        print(sys.argv)

    def do_exit(self, line):
        """Exit"""
        return True

    def do_EOF(self, line):
        """Exit"""
        return True

    def postloop(self):
        print("Done.")


def main():
    # Command line parsing
    parser = argparse.ArgumentParser(prog="DataMan", add_help=False, usage='''
    dm <command> [<args>]

    Currently implemented commands:
        cli     Interactive CLI
        ls      Basic target statistics
        vis     Simple data visualizer
        ''')
    parser.add_argument('command', help='Command to execute', nargs='?', default=None)
    parser.add_argument('-d', '--debug', action='store_true',
                        help='Debug mode -- verbose output, no confirmations.')
    parser.add_argument('--version', action='version', version='%(prog)s {version}'.format(version=__version__))
    parser.add_argument('-h', '--help', action='store_true', help='Show help text.')

    cli_args, cmd_args = parser.parse_known_args()

    if cli_args.command is None and cli_args.help:
        parser.print_help()
        sys.exit(0)

    if cli_args.debug:
        NO_EXIT_CONFIRMATION = True

    # we need to re-append arguments that should go down the rabbit hole
    if cli_args.help:
        cmd_args.append('-h')
    if cli_args.debug:
        cmd_args.append('-d')

    logging.addLevelName(LOG_LEVEL_VERBOSE, 'VERBOSE')
    log_level = LOG_LEVEL_VERBOSE if cli_args.debug else LOG_LEVEL
    logging.basicConfig(level=log_level,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    logger.debug('CLI_ARGS: {}'.format(cli_args))
    logger.debug('CMD_ARGS: {}'.format(cmd_args))

    # start cli
    if cli_args.command in [None, 'cli']:
        logger.debug('Starting CLI via command: {}'.format(cli_args.command))
        try:
            DataMan().cmdloop()
        except KeyboardInterrupt:
            pass

    # some other command was given
    else:
        logger.debug('Command {}, args: {:}'.format(cli_args.command, ' '.join(cmd_args)))
        DataMan().onecmd('{} {}'.format(cli_args.command, ' '.join(cmd_args)))


if __name__ == "__main__":
    main()
