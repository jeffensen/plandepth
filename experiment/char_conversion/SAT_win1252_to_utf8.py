#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 17:58:14 2020

@author: sascha
"""

import os, sys, codecs, getopt

def conv1252_UTF8(filename_in, filename_out):

    with codecs.open(filename_in, 'r', 'windows-1252') as fin:
        with codecs.open(filename_out, 'w', 'utf-8') as fout:
            for line in fin:
                fout.write(line)


def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], "ho:v", ["help", "output="])
    except getopt.GetoptError as err:
        # print help information and exit:
        print(err) # will print something like "option -a not recognized"
        usage()
        sys.exit(2)
    output = None
    verbose = False


    for fl in args:
        if ".m" in fl and "~" not in fl:
            print("Converting %s to UTF-8"%fl)
            conv1252_UTF8(fl, "%s_utf8"%fl)

if __name__ == "__main__":
    main()