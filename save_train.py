#!/usr/bin/env python3
import sys

#sys.stderr.write("GTP engine ready\n")
#sys.stderr.flush()
import os

def make_folder(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)

def save_txt(folder,name,text,):
    b = folder
    xxoo = b+'/'+ name + '.sgf'

    file = open(xxoo,'w')

    file.write(text)

    file.close()
    print ('ok')

