#!/usr/bin/env python

from __future__ import print_function
import tools
import os
from termcolor import colored

EXT_VIDEO = ['.avi', '.mp4', '.mkv', '.wmv']
EXT_SOUND = ['.wav', '.mp3', '.snd', '.wma']
EXT_IMAGE = ['.png', '.bmp', '.jpg', '.jpeg', '.pgm']
EXT_DOC = ['.md', '.toml', '.xml', '.tsv', '.csv', '.txt', '.doc', '.rst']

table_hdr = "{0:^28}{sep}{1:^6}{sep}{2:>3}{sep}{3:>3}{sep}{4:>3}{sep}{5:>3}{sep}{6:^10}{sep}".format(
"Folder name", "size", "#fil", "#vid", "#img", "#snd", "format", sep="|")

_row = "{0:<28}{1}{2:>4}{3:>4}{4:>4}{5:>4}{6:>10}"

def check_format(*targets):
    """Check if directory or list of files contains a dataset of known format (OE, Kwik, etc.)"""
    if len(targets) == 1 and os.path.isdir(targets[0]):
        root, dirs, files = next(os.walk(targets[0])) 
    else:
    #    for t in targets:
# TODO            assert(os.path.exists(t))
        files = targets

    for f in files:
        if fext(f) in ['.continuous']:
            return "OpenEphys"
        elif fext(f) in ['.kwx', '.kwd', '.kwik']:
            return "Kwik"
    else:
        return None
def fext(fname):
    return os.path.splitext(fname)[1]

def dir_details(path):
    name = path
    size = tools.dir_size(path)
    root, dirs, files = next(os.walk(path))
    num_files = len(files)
    num_vid = len([f for f in files if fext(f) in EXT_VIDEO])
    num_img= len([f for f in files if fext(f) in EXT_SOUND])
    num_snd = len([f for f in files if fext(f) in EXT_IMAGE])
    num_doc = len([f for f in files if fext(f) in EXT_DOC])
    data_fmt = check_format(*files)
    return dict(fname=name,
            size=size,
            num_files=num_files,
            num_vid=num_vid,
            num_img=num_img,
            num_snd=num_snd,
            num_doc=num_doc,
            data_fmt=data_fmt)

def gather(path):
    root, dirs, files = next(os.walk(path))
    details = [dir_details(root)]

    if check_format(root):
        return details
    else:
        for d in dirs:
            details.append(dir_details(d))
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

def mk_row(row, colorized=True, cols=['fname', 'size', 'num_files',
                                         'num_vid', 'num_img', 'num_snd',
                                         'data_fmt'], sepr='|'):
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
        elif c == 'num_vid':
            row_str += prettify(row[c], 
                    color='green' if row[c]>0 and colored else None,
                    sepr=sepr, align='>', width=4)
        elif c == 'num_img':
            row_str += prettify(row[c], 
                    color='green' if row[c]>0 and colored else None,
                    sepr=sepr, align='>', width=4)
        elif c == 'num_snd':
            row_str += prettify(row[c], 
                    color='green' if row[c]>0 and colored else None,
                    sepr=sepr, align='>', width=4)
        elif c == 'data_fmt':
            if row[c] == 'OpenEphys':
                color = 'yellow'
            elif row[c] == 'Kwik':
                color = 'green'
            else:
                color = 'red'
            row_str += prettify(row[c], 
                    color=color if colored else None,
                    sepr=sepr, align='>', width=10)
        else:
            row_str += prettify(row[c])

    return row_str

if __name__ == "__main__":
    color = True
    print(table_hdr)
    for row in gather(".")[:-9]:
        print(mk_row(row))
