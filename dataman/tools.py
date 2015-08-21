#!/usr/bin/env python

import os
from os.path import join, getsize
from termcolor import colored

def fmt_size(num, unit='B', si=True, sep=' ', col=False, pad=0):
    colors = {"k": "blue", "M": "green", "G": "red", "T": "cyan",
              "Ki": "blue", "Mi": "green", "Gi": "red", "Ti": "cyan"}
    if si:
        prefixes = ['', 'k', 'M', 'G', 'T', 'P', 'E']
    else:
        prefixes = ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei']
    
    divisor = 1000 if si else 1024
    for prefix in prefixes:
        if abs(num) < divisor:
            if col:
                prefix = colored(prefix, colors[prefix]) if prefix else ' '
            return "{:5.1f}{}{}{}".format(num, sep, prefix, unit, pad=pad-6)
        num /= divisor

def directory_content(path):
    return next(os.walk(path))

def dir_size(path):
    total_size = 0
    for root, dirs, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            total_size += os.path.getsize(fp)
    return total_size

def stats(path):
    print "Got path:", path
    root, dirs, files = directory_content(path)
    print root, "consumes",
    print format_filesize(sum(getsize(join(root, name)) for name in files)),
    print "in", len(files), "non-directory files"
    print "Directories:\n"
    for d in dirs:
        print d, fmt_size(dir_size(d))
    print "Files:\n", files

if __name__ == "__main__":
    stats('.')
    print fmt_size(dir_size("."))
