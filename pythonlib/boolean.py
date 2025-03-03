#!/usr/bin/env python
# coding: utf-8
#引入numpy 库
#import matplotlib.pyplot as plt
import numpy as np

#定义布尔值之间运算
def otimes_b_b (a,b):
    if a==b:
        return True
    else:
        return False
#布尔向量定义为python中的列表 list
#定义布尔向量与布尔值之间计算
def otimes_bv_b(bv,b):
    return [otimes_b_b(x,b) for x in bv]
#定义布尔向量之间运算
def otimes_bv_bv(bv1,bv2):
    temp=[otimes_b_b(x,y) for x,y in zip(bv1,bv2)]
    #如果两个向量之间相同位置布尔值相同的部分大于不同的部分
    if temp.count(True)>temp.count(False):
        #返回 True
        return True
    else:
        #否则返回 False
        return False
#矩阵定义为numpy中的array
#定义布尔矩阵之间运算
def otimes_bm_bm(bm1,bm2):
    #后一个矩阵列向量变为行向量
    bm2_t=bm2.T
    temp =[otimes_bv_bv(x,y) for x in bm1
              for y in bm2_t]
    temp =np.array(temp)  
    return temp.reshape((bm1.shape[0],bm2.shape[1]))
#定义矩阵与向量之间运算
def otimes_bm_bv(bm,bv):
    temp=[otimes_bv_bv(x,bv) for x in bm]
    return temp
#定义矩阵与向量之间运算
def otimes_bv_bm(bv,bm):
    temp=[otimes_bv_bv(x,bv) for x in bm]
    return temp    
#定义矩阵与布尔值之间计算
def otimes_bm_b(bm,b):
    temp=[otimes_bv_b(x,b) for x in bm]
    return np.array(temp)




def  qtimes_bv_bv(bv1,bv2):
    temp =[otimes_b_b(x,y) for x,y in zip(bv1,bv2)]
 
    return temp   

def qtimes_bv_bm(bv1,bm1):
    temp=[qtimes_bv_bv(bv1,x) for x in (list(bm1))]
    return np.array(temp)

def rand_bv(n):
    a=np.random.rand(1,n)
    x=a>0.5
    return x[0].tolist()  


def qtimes_loss(bv1,bv2):
    temp = qtimes_bv_bv(bv1,bv2)
    return temp.count(False)/len(temp)


def perfect_match_bv_bv(bv1,bv2):
	if bv1==bv2:
		return True
	else:
		return False

def perfect_match_bv_bm(bv1,bm1):
    temp=[perfect_match_bv_bv(bv1,list(x)) for x in (list(bm1))]
    return temp


def convolution(gfun, oa, ob):
    #n=0
    res=[]
    oa_rows=oa.shape[0]
    oa_cols=oa.shape[1]
    ob_rows=ob.shape[0]
    ob_cols=ob.shape[1]
    for i in range (0,(oa_rows-ob_rows+1)):
        #print (i)
        for j in range (0,(oa_cols-ob_cols+1)):
            temp=((oa[i:(i+ob_rows),:])[:,j:(j+ob_cols)])
            tempx=temp.reshape(1,temp.shape[0]*temp.shape[1]).tolist()[0]
            tempy=ob.reshape(1,ob.shape[0]*ob.shape[1]).tolist()[0]
            res_temp=gfun(tempx,tempy)
            res.append(res_temp)
            #print(b)
            #show_nparray(b)
            #n=n+1
    res=np.array(res)
    #print(n)

    return res.reshape((oa_rows-ob_rows+1),(oa_cols-ob_cols+1))  