from ssm import StateSpaceModel
from ssm import LinearModelParameters
#from SLDS import SLDS
from simulation import Simulation
from ssm import LGSSM
import numpy as np
import copy
from utils import dec_to_base
import matplotlib.pyplot as plt

tens = np.random.rand(2,2,2,2)
print(tens)
red_tail_idx = 10
num_models = 5
r = 3
i_r = 0
li = list(map(int, list(dec_to_base(red_tail_idx, num_models).zfill(r-1))))#.append(10)
newli = copy.copy(li)
newli.append(3)
print("li=",li)
print("newli=", newli)
newli = copy.copy(li)
newli.insert(0,4)
print("li=",li)
print("newli=", newli)
tail_list = tuple(newli)
#print(np.shape(tail_list))
#tail_list = np.reshape(tail_list, (r, 1))
#print(np.shape(tail_list))
print(tens[tail_list])

ind = (1, 1)
print(tens[0])
