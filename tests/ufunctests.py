import os
import sys

up = os.path.normpath(os.path.join(os.getcwd(), "../src"))
sys.path.append(up)
sys.path.append(os.getcwd())

import numpy as np
import torch as tc
from indexedtensor import tensor, zeros

tc.set_default_dtype(tc.float64)

A = tensor(tc.rand(4, 4))
B = tensor(tc.rand(2, 3, 4))
b = tensor(tc.rand(3, 4))
x = tensor(tc.rand(4))
y = tensor(tc.rand(3))


z = tc.sin(x.i)
print(z._indices_)
