import os
import sys

up = os.path.normpath(os.path.join(os.getcwd(), "../src"))
sys.path.append(up)
sys.path.append(os.getcwd())

import numpy as np
from tensors import tensor, zeros

A = tensor(np.random.rand(4, 4))
B = tensor(np.random.rand(2, 3, 4))
b = tensor(np.random.rand(3, 4))
x = tensor(np.random.rand(4))
y = tensor(np.random.rand(3))


def test(a, b, f=np.array_equal):
    assert f(a, b)


close = lambda a, b: np.allclose(a, b, 1e-10, 1e-10)

test(A.ii._, np.trace(A))
test(+A.ii, np.trace(A))

test(A.ii.i, np.diagonal(A))

# use einsum rather than np.sum because results *do* differ by 1e-15
test(A.ij.i, np.einsum("ij->i", A))
test(A.ij.j, np.einsum("ij->j", A))
test(A.ij._, np.einsum("ij->", A))

test(A.ij.ji, np.transpose(A))
test(B.ijk.jki, np.transpose(B, (1,2,0))) 

test(+(x.i * x.i), np.dot(x, x), close)
test((x.i * y.j).ij, np.outer(x, y), close)

test(+(A.ij * x.j), A @ x)
test((A.ij * x.j).i, A @ x)

test((B.ijk * b.jk).i, np.tensordot(B, b, 2), close)

test((A.ij * x.j).ij, A * x, close)
test((A.ij * x.i).ij, A * x.reshape((4, 1)), close)

# addition

test((x.i + y.j).ij, x.reshape((4, 1)) + y)
test((A.ij + x.j).ij, A + x)
test((A.ij + x.i).ij, A + x.reshape((4, 1)))
test((A.ij + x.k).ijk, A.reshape((4, 4, 1)) + x)
test((x.i + zeros(4).j).ij, x.reshape((4, 1)) + np.zeros(4))
test((A.ij + x.i).i, np.sum(A + x.reshape((4, 1)), axis=1), close)

test((A.ij * x.j + x.i).i, A @ x + x)

x.index('i')