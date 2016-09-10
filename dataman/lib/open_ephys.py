#!/usr/bin/env python
# -*- coding: utf-8 -*-

# from __future__ import print_function
import os
import xml.etree.ElementTree as etree
from dataman.lib.tools import fext, dir_content


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
            fv = format_version(root, dirs, files)
            return "OE_v{}".format(fv if fv else '???')
    else:
        return False


def format_version(root, dirs=None, files=None):
    if dirs is None or files is None:
        _, dirs, files = dir_content(root)
    if "settings.xml" in files:
        root = etree.parse(os.path.join(root, 'settings.xml'))
        version = root.findall("INFO/VERSION")
        if not len(version):
            return None
        else:
            return version[0].text
    else:
        return None
