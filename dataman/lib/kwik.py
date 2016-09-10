#!/usr/bin/env python
# -*- coding: utf-8 -*-

# from __future__ import print_function
import os
import xml.etree.ElementTree as etree
from dataman.lib.tools import fext, dir_content


def detect(root, dirs=None, files=None):
    for f in files:
        if fext(f) in ['.kwx', '.kwd', '.kwik']:
            fv = format_version(root, dirs, files)
            return "Kw_v{}".format(fv if fv else '???')

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
