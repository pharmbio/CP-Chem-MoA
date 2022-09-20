import numpy as np
import six

from chainer import cuda
from chainer import function
from chainer.functions.activation import sigmoid
from chainer import utils
from chainer.utils import type_check  

#-------------------------------------------------------------
 #Sigmoid function definition
def strong_sigmoid(x):
    return 1*(x >=0)

#-------------------------------------------------------------
 #detaset function definition
def random_list(x, seed=0):
    np.random.seed(seed)
    np.random.shuffle(x)

def data_boostDataset(P,N,boost=1,seed=0):
    random_list(P,seed)
    random_list(N,seed)
    T = [0]*len(N)+ [1]*(len(P)*boost)
    for i in range(boost):N.extend(P)
    random_list(N,seed)
    random_list(T,seed)
    return N, T
