#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import cmd
import logging
import os
import os.path as op
import sys
import subprocess
import shlex

from .lib.constants import LOG_LEVEL_VERBOSE

__version__ = '0.1.5'

current_path = os.getcwd()
try:
    os.chdir(op.dirname(__file__))
    GIT_VERSION = subprocess.check_output(["git", "describe", "--always"]).strip().decode('utf-8')
except subprocess.CalledProcessError as e:
    GIT_VERSION = "Unknown"
os.chdir(current_path)

NO_EXIT_CONFIRMATION = True
LOG_LEVEL = logging.INFO
log_level = LOG_LEVEL


class DataMan(cmd.Cmd):
    """Command line tool for quick data documentation."""

    prompt = "dm> "
    intro = "+ DataMan is here to help! +"
    intro = '\n'.join(["="*len(intro), intro, "="*len(intro)])

    log = logging.getLogger(__name__)

    def preloop(self):
        self.log.debug("Starting DataMan CLI")

    def do_ls(self, args_string):
        parser = argparse.ArgumentParser('Recording statistics',)
        parser.add_argument('path', help='Relative or absolute path to directory',
                            default='.', nargs='?')
        parser.add_argument('-d', '--debug', action='store_true',
                            help='Debug mode -- verbose output, no confirmations.')

        cli_args = parser.parse_args(args_string.split(' ') if args_string else '')
        self.log.debug('ls with args: {}'.format(cli_args))
        path = op.abspath(op.expanduser(cli_args.path))
        self.log.debug('Expanded path: {}'.format(path))

        import dataman.dirstats.dirstats as ds
        ds.print_table(ds.gather(path))

    def do_stats(self, args_string):
        self.do_ls(args_string)
        # parser = argparse.ArgumentParser('Recording statistics')
        # parser.add_argument('path', help='Relative or absolute path to directory',
        #                     default='.', nargs='?')
        # parser.add_argument('-d', '--debug', action='store_true',
        #                     help='Debug mode -- verbose output, no confirmations.')
        # cli_args = parser.parse_args(line.split(' ') if line else '')
        # self.log.debug('Stats with args: {}'.format(cli_args))
        # path = op.abspath(op.expanduser(cli_args.path))
        # self.log.debug('Expanded path: {}'.format(path))
        #
        # import dataman.dirstats.dirstats as ds
        # ds.print_table(ds.gather(path))

    def do_vis(self, args_string):
        from dataman.vis import vis
        self.log.info('Starting visualization with dataman v{} @git [{}]'.format(__version__, GIT_VERSION))
        vis.run(args_string.split(' '))

    def do_conv(self, args_string):
        from dataman.conv import convert
        self.log.info('Starting conversion with dataman v{} @git [{}]'.format(__version__, GIT_VERSION))
        convert.main(args_string.split(' '))

    def do_check(self, args_string):
        pass

    def do_proc(self, args_string):
        print(sys.argv)

    def do_exit(self, args_string):
        """Exit"""
        return True

    def do_EOF(self, args_string):
        """Exit"""
        return True

    def postloop(self):
        print("Done.")


def main():
    # Command line parsing
    # FIXME: Stupid spaces-in-file-names issues... especially after the string conversion to onecmd

    parser = argparse.ArgumentParser(prog="DataMan", add_help=False, usage='''
    dm <command> [<args>]

    Currently implemented commands:
        cli     Interactive CLI
        ls      Basic target statistics
        vis     Simple data visualizer
        conv    Convert formats and layouts
        ''')
    parser.add_argument('command', help='Command to execute', nargs='?', default=None)
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Debug mode -- verbose output, no confirmations.')
    parser.add_argument('--version', action='version', version='%(prog)s {version}'.format(version=__version__))
    parser.add_argument('-h', '--help', action='store_true', help='Show help text.')

    cli_args, cmd_args = parser.parse_known_args()

    if cli_args.command is None and cli_args.help:
        parser.print_help()
        sys.exit(0)

    if cli_args.verbose:
        NO_EXIT_CONFIRMATION = True

    # we need to re-append arguments that should go down the rabbit hole
    if cli_args.help:
        cmd_args.append('-h')
    # if cli_args.verbose:
    #     cmd_args.append('-v')

    logging.addLevelName(LOG_LEVEL_VERBOSE, 'VERBOSE')
    log_level = LOG_LEVEL_VERBOSE if cli_args.verbose else LOG_LEVEL
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
