from . import boolean as boolean

import numpy as np
import sqlite3 as sqlite3

class bool_convnet:
    def __init__(self, dbname, ci ,cc):
        self.ci=ci
        self.cc=cc
        #self.con = sqlite3.connect(dbname)
        self.inode=(np.random.rand(ci,ci))>0
        self.cnode=(np.random.rand(cc,cc))>0
        self.onode=(np.random.rand(ci,ci))>0

    def feedforward(self):
        temp=np.pad(self.inode,int((self.cc-1)/2),mode='constant')
        self.onode=boolean.convolution(boolean.otimes_bv_bv,temp,self.cnode)


