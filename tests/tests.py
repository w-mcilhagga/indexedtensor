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


def test(a, b, f=np.array_equal):
    assert f(a, b)


close = lambda a, b: tc.allclose(a, b, 1e-10, 1e-10)

test(A.ii._, tc.trace(A))
test(+A.ii, tc.trace(A))

test(A.ii.i, tc.diagonal(A))
test(A["ii->i"], tc.diagonal(A))

# use einsum rather than sum because results *do* differ by 1e-15
test(A.ij.i, tc.einsum("ij->i", A))
test(A.ij.j, tc.einsum("ij->j", A))
test(A.ij._, tc.einsum("ij->", A))

test(A.ij.ji, tc.transpose(A, 0, 1))
test(B.ijk.jki, tc.permute(B, (1, 2, 0)))

test(+(x.i * x.i), tc.dot(x, x), close)
test((x.i * y.j).ij, tc.outer(x, y), close)

test(+(A.ij * x.j), A @ x, close)
test((A.ij * x.j).i, A @ x, close)

test(+(x.i * x.j * A.ij), tc.einsum("i,j,ij->", x, x, A))

test((B.ijk * b.jk).i, tc.tensordot(B, b, 2), close)
test(+(B.ijk * b.jk), tc.tensordot(B, b, 2), close)

test((A.ij * x.j).ij, A * x, close)
test((A.ij * x.i).ij, A * tc.reshape(x, (4, 1)), close)

test(+(x.i + y.j), x.reshape((4, 1)) + y)
test((x.i + y.j).ij, x.reshape((4, 1)) + y)

test((A.ij + x.j).ij, A + x)
test((A.ij + x.i).ij, A + x.reshape((4, 1)))

test((A.ij + x.k).ijk, A.reshape((4, 4, 1)) + x)
test(+(A.ij + x.k), A.reshape((4, 4, 1)) + x)

test((x.i + zeros(4).j).ij, x.reshape((4, 1)) + tc.zeros(4), close)
test((A.ij + x.i).i, tc.sum(A + x.reshape((4, 1)), axis=1), close)

test((A.ij * x.j + x.i).i, A @ x + x, close)
test(+(A.ij * x.j + x.i), A @ x + x, close)

print("all tests passed")


def itensorwrap(f):
    # makes first arg an indexed tensor
    def ff(t, *args, **kwargs):
        return f(tensor(t), *args, **kwargs)

    return ff


@itensorwrap
def qform(t):
    return +(t.i * t.j * A.ij)


from torch.autograd.functional import jacobian, hessian

g = hessian(qform, tensor([1, 2, 3.0, 4]))
