from . import boolean as boolean

import numpy as np
import sqlite3 as sqlite3

class endecoder_net:
    def __init__(self, dbname, ci ,co):
        #self.con = sqlite3.connect(dbname)
        self.inode=((np.random.rand(1,ci))>0)[0].tolist()
        self.onode=((np.random.rand(1,co))>0)[0].tolist()
        self.wi=((np.random.rand(co,ci))>0.5)
    def feedforward(self):
        self.onode=boolean.otimes_bv_bm(self.inode,self.wi)
    def feedbackward(self):
        self.inode=boolean.otimes_bv_bm(self.onode,self.wi.T)
    def backPropagate(self, targets):        
        #calculate error
        self.onode_error=boolean.qtimes_bv_bv(targets,self.onode)
        #self.wo=boolean.otimes_bv_bm(error,self.wo)
        #self.hnodetargets=boolean.otimes_bv_bm(self.error, np.transpose(self.wo))
        #self.hnode_error=[boolean.otimes_b_b(x,y) for x,y in zip(self.hnodetargets,self.hnode)]
        #modify out weight matrix
        self.wi=(boolean.qtimes_bv_bm(self.onode_error,self.wi.T)).T

        #self.inode=boolean.otimes_bv_bm(self.onode,self.wi.T)
    def forwardPropagate(self, targets):        
        #calculate error
        self.inode_error=boolean.qtimes_bv_bv(targets,self.inode)

        #modify in weight matrix
        self.wi=(boolean.qtimes_bv_bm(self.inode_error,self.wi))

        #self.onode=boolean.otimes_bv_bm(self.inode,self.wo)

    def train(self,inputs,targets):
            #
        inputs_flip=[not(x) for x in inputs]
        targets_flip=[not(x) for x in targets]
        #
        self.inode=inputs
        self.feedforward()
        self.backPropagate(targets)
##
        self.onode=targets_flip
        self.feedbackward()
        self.forwardPropagate(inputs_flip)
#

        #
        self.inode=inputs
        self.feedforward()




