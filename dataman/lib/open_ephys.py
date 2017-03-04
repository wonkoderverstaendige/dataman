#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import xml.etree.ElementTree as ET
from .tools import fext, dir_content


def detect(root=None, dirs=None, files=None):
    """Checks for existence of an open ephys formatted data set in the root directory.

    Args:
        root: Directory to search in.
        dirs: list of subdirectories in root. Will be scanned if not provided.
        files: List of files in the root directory. Will be scanned if not provided.

    Returns:
        None if no data set found, else a string with data set format name and version.
    """
    # TODO: Make all three optional and work with either
    if dirs is None or files is None:
        _, dirs, files = dir_content(root)

    for f in files:
        if fext(f) in ['.continuous']:
            settings_xml = _find_settings_xml(root)
            if settings_xml is None:
                fv = None
            else:
                fv = format_version(settings_xml)
                # print(_fpga_node(settings_xml))
            return "OE_v{}".format(fv if fv else '???')
    else:
        return False

def _fpga_node(path):
    chain = _config(path)['SIGNALCHAIN']
    nodes = [p['attrib']['NodeId'] for p in chain if p['type']=='PROCESSOR' and 'FPGA' in p['attrib']['name']]
    if len(nodes) == 1:
        return nodes[0]
    else:
        raise BaseException('Node ID not found in xml file {}'.format(path))


def format_version(path):
        settings = _config(path)
        return settings['INFO']['VERSION']


def _find_settings_xml(base, dirs=None, files=None):
    if dirs is None or files is None:
        _, dirs, files = dir_content(base)
    if "settings.xml" in files:
        return os.path.join(base, 'settings.xml')
    else:
        return None


def _config(path):
    """Reads Open Ephys XML settings file and returns dictionary with relevant information.
        - Info field
            Dict(GUI version, date, OS, machine),
        - Signal chain
            List(Processor or Switch dict(name, nodeID). Signal chain is returned in ORDER OF OCCURRENCE
            in the xml file, this represents the order in the signal chain. The nodeID does NOT reflect this order, but
            but the order of assembling the chain.
        - Audio
            Int bufferSize

    Args:
        path: Path to settings.xml file

    Returns:
        Dict{INFO, SIGNALCHAIN, AUDIO}
    """
    root = ET.parse(path).getroot()
    info = dict(
        VERSION = root.find('INFO/VERSION').text,
        DATE = root.find('INFO/DATE').text,
        OS = root.find('INFO/OS').text,
        MACHINE = root.find('INFO/VERSION').text
    )

    sc = root.find('SIGNALCHAIN')
    chain = [dict(type=e.tag, attrib=e.attrib) for e in sc.getchildren()]

    audio = root.find('AUDIO').attrib
    return dict(INFO=info,
                SIGNALCHAIN=chain,
                AUDIO=audio)
