#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import sys
from termcolor import colored
from . import tools, open_ephys, kwik
from .tools import fext, dir_content

EXT_VIDEO = ['.avi', '.mp4', '.mkv', '.wmv']
EXT_SOUND = ['.wav', '.mp3', '.snd', '.wma']
EXT_IMAGE = ['.png', '.bmp', '.jpg', '.jpeg', '.pgm']
EXT_DOC = ['.md', '.toml', '.xml', '.tsv', '.csv', '.txt', '.doc', '.rst']

DEFAULT_COLS = ['fname', 'size', 'num_files', 'num_vid', 'num_img', 'num_snd', 'num_doc', 'data_fmt']
COLUMNS = {'fname': ['F']}

class Column:
    def __init__(self, name, width=3, fmt=':', align='^', *args, **kwargs):
        self.name = name
        self.w = width
        self.align = align
        self.fmt = '{'+fmt+align+str(width)+'}'

    def header(self):
        if self.name in COLUMNS:
            return self.fmt.format(COLUMNS[self.name])
        else:
            return self.fmt.format('Unknown')

    def row(self, data):
        return self.fmt.format(data)

table_hdr = "{:^28}{sep}{:^6}{sep}{:>3}{sep}{:>3}{sep}{:>3}{sep}{:>3}{sep}{:>3}{sep}{:^10}{sep}".format(
"Folder name", "size", "#fil", "#vid", "#img", "#snd", '#doc', "format", sep=" ")

_row = "{0:<28}{1}{2:>4}{3:>4}{4:>4}{5:>4}{6:>10}"

def contains_data(root, dirs=None, files=None):
    """Check if directory or list of files contains a dataset of known format (OE, Kwik, etc.)"""
    if None in [dirs, files]:
        _, dirs, files = dir_content(root)

    formats = [open_ephys, kwik]
    for fmt in formats:
        detected = fmt.detect(root, dirs, files)
        if detected:
            return detected
    else:
        return None

def dir_details(path):
    root, dirs, files = dir_content(path)

    name = os.path.basename(path)
    size = tools.dir_size(path)
    num_files = len(files)
    num_vid = len([f for f in files if fext(f) in EXT_VIDEO])
    num_img= len([f for f in files if fext(f) in EXT_IMAGE])
    num_snd = len([f for f in files if fext(f) in EXT_SOUND])
    num_doc = len([f for f in files if fext(f) in EXT_DOC])
    data_fmt = contains_data(path)

    return dict(fname=name, size=size, num_files=num_files, num_vid=num_vid,
            num_img=num_img, num_snd=num_snd, num_doc=num_doc,
            data_fmt=data_fmt)

def data_stats(path):
    fmt = contains_data(path)
    assert(fmt)


def gather(path):
    """Gather details on the path and its subdirectories.

    Args:
        path: Relative or absolute path to a directory.

    Returns:
        List of dictionaries. Each element in the list corresponds
        to the details of a single directory (including the given as
        [path]) in a dictionary.
    """
    root, dirs, files = dir_content(path) 

    details = []
    details.append(dir_details(root))
    for d in dirs:
        details.append(dir_details(os.path.join(root, d)))
    return details
        
def prettify(element, color=None, align='>', width=0, sepl='', sepr=''):
    text = "{:{align}{width}}".format(element, align=align, width=width)
    if color:
        return sepl + colored(text, color) + sepr
    else:
        return sepl+text+sepr

def fit_str(string, max_len=10, weight=0.7):
    if len(string) < max_len or max_len < 4:
        return string
    
    indicator = '[..]'
    head = int((max_len-len(indicator))*(1-weight))
    tail = int((max_len-len(indicator))*weight)
    return string[:head]+indicator+string[-tail:]

def mk_row(row, colorized=True, cols=DEFAULT_COLS, sepr='|'):
    row_str = ''
    for c in cols:
        if c == 'fname':
            row_str += prettify(fit_str(row[c], 28), sepr=sepr, align='<', width='28')
        elif c == 'size':
            row_str += prettify(tools.fmt_size(row[c], unit='', sep='', col=True, pad=7),

                    sepr=sepr, align='>', width='')

        elif c == 'num_files':
            row_str += prettify(row[c], 
                    color='red' if row[c]==0 and colored else None,
                    sepr=sepr, align='>', width=4)

        elif c in ['num_vid', 'num_img', 'num_snd', 'num_doc']:
            if row[c] > 0:
                color='green' if colored else None
                val = row[c]
            else:
                val = ''
                color = None
            row_str += prettify(val, color=color, sepr=sepr, align='>', width=4)

        elif c == 'data_fmt':
            if row[c] is None:
                color = None
            else:
                if 'OE' in row[c]:
                    color = 'yellow'
                elif 'Kw' in row[c]:
                    color = 'green'
                else:
                    color = None
            row_str += prettify(row[c] if row[c] is not None else '', 
                    color=color if colored else None,
                    sepr=sepr, align='>', width=10)
        else:
            row_str += prettify(row[c])

    return row_str


getch = tools._find_getch()
def print_table(rows, color=True, page_size=-1):
    termh, termw = tools.terminal_size()
    if page_size is not None and page_size < 1:
        page_size = termh - 5 
    line_length = None
    for i, row in enumerate(rows):
        row_string = mk_row(row)
        if line_length is None:
            line_length = len(tools.strip_ansi(row_string)) 

        # pause after printing full page of rows
        if page_size is not None and page_size > 1 and i%(page_size+1) == 0:
            if i > 1:
                print("[MORE]")
                c = getch()
                sys.stdout.write("\033[F")
                if c == 'q':
                    print('\n ...{} not shown.'.format(len(rows)-i))
                    break
            print(table_hdr)
            print('_'*line_length)

        # print current line
        print(row_string)
 
if __name__ == "__main__":
    print_table(gather('.'))    

