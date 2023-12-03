#!/usr/bin/env python
# coding: utf-8

# # Indexed tensors.
# 
# The `Tensor` class creates a subclass of numpy array which (ab)uses attribute lookup to define the axes of the tensor. 
# 
# If `A` is a 3D tensor, then `A.ijk` is an **indexed tensor** which labels the three axes `i`, `j`, and `k` respectively. The number of letters in the index must match the number of axes in the tensor, except when double-underscores (dunders) are used. `A.i__` with a dunder says the first axis is labelled `i` and the others have no label, however many there are. `A.__i` says the last axis is labelled `i` and the others have no label, however many there are, and `A.i__j` labels the first and last axes as `i` and `j`.
# 
# An indexed tensor is a view of the original tensor.

# Indexed tensors can be multiplied together. For example `A.ij*x.j` represents a tensor whose `[i,j]`-th entry is `A[i,j]*x[j]`. However, this
# computation is not carried out until a summation is declared. Summation is indicated by **another** indexing attribute which gives the indices/axes that are **not** summed over.
# 
# Some examples:
# 
# * `(A.ij*x.j).j` sums the tensor `A.ij*x.j` over the axis `i` (the first axis) to give a 1D tensor $t_j = \sum_i{A_{ij}x_j}$. The brackets around `A.ij*x.j` are needed because the
#   attribute operator `.` binds tightly
#   
# * `(A.ij*x.j).i` sums the tensor over the axis `j` (the second axis) to give a 1D tensor $t_i = \sum_j{A_{ij}x_j}$. This is standard matrix-vector multiplication
#   
# * `(A.ij*x.i)._` sums the tensor over the axes `i` and `j` to give $t = \sum_{i,j}{A_{ij}x_j}$. A single underscore index says to sum over all axes.
#   
# * Finally, `(A.ij*x.i).ij` does no summation and yields a 2D tensor with elements $t_{ij} = A_{ij}x_j$
# 
# Any time an indexed expression like `A.ij*x.j` is summed, it yields an unindexed tensor, which can be indexed again as needed.
# 
# The Einstein summation convention is that any repeated indices are summed over. This convention is implemented through the unary plus operator. For example,<br/> `+(A.ij*x.j)` will sum over the repeated index `j`, and is equivalent to `(A.ij*x.j).i`. 

# The multiplication and indexing replicates the functionality of the numpy einsum function, because it is implemented using it. The following operations can be performed with this notation:
# 
# * Trace: `A.ii._` or `+A.ii`
# * Diagonal: `A.ii.i`
# * Summation over one axis: `A.ij.i` or `A.ij.j`, equivalent to `np.sum(A, axis=1)` or `np.sum(A, axis=0)`
# * Summation over all axes: `A.ijk._` is the equivalent of `np.sum(A)`
# * Transpose: `A.ij.ji`
# * Move axis: `A.ijk.jki`
# * Inner (dot) product: `(x.i*x.i)._` or `+(x.i*x.i)`
# * Outer product: `(a.i*b.j).ij`
# * Matrix multiplication: `(A.ij*x.j).i` or `+(A.ij*x.j)`
# * Quadratic form: `(x.i*A.ij*x.j)._` or `(x.i*x.j*A.ij)._` or `+(x.i*A.ij*x.j)`(NB indexed multiplication is commutative)
# * Tensor contractions of all types: `(A.ijk*b.jk).i`, `(A.ijk*b.ik).j`, or `+(A.ijk*b.ik)` etc.
# * Broadcasting: `(A.ij*x.i).ij` over the first axis or `(A.ij*x.j).ij` over the second
# 
# `np.einsum` also lets you use ellipses in summation formulas. In the tensor class, a dunder index `__` is used to represent ellipses.
# 
# In all of these cases, the result is an unindexed tensor.

# ## Interoperability
# 
# Tensors, either indexed or bare, interact smoothly with numpy arrays, functions and scalars. Indexed tensors keep their indices.
# 
# For example:
# 
# * `A.ij+5` returns an indexed tensor (with indices `ij`) where every element has been increased by 5. `A+5` would retunr an unindexed tensor.
# 
# * `A.ij*B` does standard element-by-element multiplication of the indexed tensor `A` and the unindexed tensor or ndarray `B`.

# ## Tensor Broadcasting
# 
# The Tensor class also implements indexed addition, or any mix of indexed addition and multiplication.
# 
# Simple cases - no summation
# ```python
# (A.ij+x.i).ij
# (A.ij+x.j).ij
# ```
# 
# some summation
# ```python
# (A.ij*b.j+x.i).i # = ((A.ij*b.j).i+x.i).i   no broadcasting occurs
# (A.ij*b.j+x.j).i # = ((A.ij*b.j).ij+x.j).i  broadcasting occurs, note that the inner .ij keeps it's indices
# ```
# 
# complex
# ```python
# ((A.ij*b.i+x.i)+y.j).i # = (((A.ij*b.i+x.i).ij+y.j).i
# ```
# 

# In[5]:


import numpy as np
from collections import Counter

def fixname(name):
    '''interprets dunder and underscores in the index name'''
    # replace __ with ellipses
    name = name.replace('__', '...')
    if name=='_':
        name = ''
    return name

def op_or_indexed(obj):
    '''True if obj is an Einop object or an indexed Tensor'''
    return isinstance(obj, Einop) or (type(obj) is Tensor and obj.indices is not None)

def implicit_indices(idx):
    '''the default summation for indices given in the string idx'''
    idx = Counter(idx)
    defidx = ''
    for e,c in idx.items():
        if c==1:
            defidx += e
    return ''.join(sorted(defidx))
            
class Tensor(np.ndarray):
    '''Tensor object is a numpy array with added einstein indices property and attribute access'''
    def __new__(subtype, shape, dtype=float, buffer=None, offset=0,
                strides=None, order=None, info=None):
        t = np.ndarray.__new__(subtype, shape, dtype, buffer, offset, strides, order)
        t.indices = None # unindexed
        return t
        
    def __array_finalize__(self, obj):
        '''needed when subclassing ndarray, adds the indices attribute'''
        if obj is not None:
            self.indices = getattr(obj, 'indices', None)
        
    def __getattr__(self, name):
        '''define indices for einstein summation/addition.
        The indices are letters (i, j, k, . . .) or _ or __.
        The double underscore behaves like ellipses
        The single underscore is used to summate over all indices.
        This returns a view of the original tensor.
        '''
        if self.indices is None:
            # we are defining the indices of the tensor
            if '...' not in name and len(name)!=self.ndim:
                # TODO: what about '...ijk' in a 2d tensor?
                raise ValueError('wrong number of indices')
            s = self.view()
            s.indices = fixname(name)
            return s
        else:
            # we have already defined the indices of the tensor and now we are
            # summing, e.g. x.ij.i sums over j
            #               x.ij._ sums over i and j
            if name==self.indices:
                self.indices = None
                return self
            else:
                return tensor(np.einsum(self.indices+'->'+fixname(name), self))

    def __pos__(self):
        '''default evaluation'''
        if self.indices is None:
            return self
        else:
            return getattr(self, implicit_indices(self.indices) )
        
    def __add__(self, other):
        '''set up an einstein-notation broadcast'''
        if self.indices is None:
            if not op_or_indexed(other):
                return np.add(self, other)
        else:
            if not isinstance(other,(Tensor, Einop)):
                return np.add(self, other)
            elif op_or_indexed(other):
                return Einadd(self, other)
        # no return occurred, so invalid
        raise RuntimeError('both indices must be defined in a Tensor addition')
        
    def __radd__(self, other):
        # commutative
        return self.__add__(other)
        
    def __mul__(self, other):
        '''set up an einstein summation.'''
        if self.indices is None:
            if not op_or_indexed(other):
                return np.multiply(self, other)
        else:
            if not isinstance(other,(Tensor, Einop)):
                return np.multiply(self, other)
            elif op_or_indexed(other):
                return Einsum(self, other)
        # no return occurred, so invalid
        raise RuntimeError('both indices must be defined in a Tensor multiplication')
    
    def __rmul__(self, other):
        # this is commutative
        return self.__mul__(other)

    
    
# creation routines

def tensor(object, *args, **kwargs):
    '''create an unindexed tensor like np.array()'''
    return np.asarray(object, *args, **kwargs).view(Tensor)

def zeros(shape):
    return tensor(np.zeros(shape))

def ones(shape):
    return tensor(np.ones(shape))


## operations. We need to add numpy functions to this.

class Einop:
    '''an object holding an unevaluated einsum/einadd'''
    def __init__(self, *operands):
        self.operands = operands

    def __add__(self, other):
        '''add an einsum object to a tensor or einsum'''
        if type(self)==Einadd:
            if type(other)==Einadd:
                return Einadd(*self.operands, *other.operands)
            if op_or_indexed(other):
                return Einadd(*self.operands, other)
        if type(self)==Einsum and op_or_indexed(other):
            return Einadd(self, other)
        raise ValueError('can only operate with indexed Tensors or Einops')
        
    def __radd__(self, other):
        '''multiply an einsum object by a tensor or einsum'''
        return self.__add__(other)

    def __mul__(self, other):
        '''multiply an einsum object by a tensor or einsum'''
        if type(self)==Einsum:
            if type(other)==Einsum:
                return Einsum(*self.operands, *other.operands)
            if op_or_indexed(other):
                return Einsum(*self.operands, other)
        if type(self)==Einadd and op_or_indexed(other):
            return Einsum(self, other)
        raise ValueError('can only operate with indexed Tensors or Einops')
        
    def __rmul__(self, other):
        '''multiply an einsum object by a tensor or einsum'''
        return self.__mul__(other)
        
class Einsum(Einop):
    '''an object holding an unevaluated einsum'''

    @property
    def indices(self):
        '''the contraction indices'''
        allindices = ''.join(map(lambda o: o.indices, self.operands))
        return implicit_indices(allindices)

    def __pos__(self):
        '''sum evaluation'''
        return getattr(self, self.indices )
    
    def __getattr__(self, name):
        '''evaluates the product'''
        script = []
        operands = []
        for op in self.operands:
            script.append(op.indices)
            if type(op)==Tensor:
                # put it in the oplist
                operands.append(op)
            else:
                # evaluate and put it in the oplist (op is an Einadd)
                operands.append(getattr(op, op.indices))
        return tensor(np.einsum(','.join(script)+'->'+fixname(name), *operands))


class Einadd(Einop):
    '''an object holding an unevaluated broadcast summation'''
    
    @property
    def indices(self):
        '''return all the unique indices'''
        return ''.join(sorted(set(''.join(map(lambda o:o.indices, self.operands)))))

    def __pos__(self):
        '''add evaluation, no contraction occurs'''
        return getattr(self, self.indices )
    
    def __getattr__(self, name):
        # index each operand by its own indices
        operands = []
        shape = {}
        for idx in self.indices:
            shape[idx]=0
        for op in self.operands:
            idx = op.indices
            op = tensor(getattr(op, idx))
            op.indices = idx
            operands.append(op)
            opshape = dict(zip(idx, op.shape))
            for i in opshape:
                shape[i] = max(shape[i], opshape[i]) 
        sum = 0.0
        for op in operands:
            b = broadcast_tensor(op, shape)
            sum += b
        return tensor(np.einsum(''.join(shape.keys())+'->'+fixname(name), sum))

def broadcast_tensor(a, shapedict):
    # broadcasts a over b
    # permute axes of a to suit ordering in b
    reorder = ''
    for i in shapedict.keys():
        if i in a.indices:
            reorder += i
    a = tensor(getattr(a, reorder))
    a.indices = reorder
    # work out new shape of a
    newshape = []
    for i in shapedict.keys():
        try:
            axis = a.indices.index(i)
            newshape.append(a.shape[axis])
        except:
            newshape.append(1)
    a = np.reshape(a, newshape)
    a = np.broadcast_to(a, [*shapedict.values()])
    return a


# ### Tests

# In[9]:


A = tensor(np.random.rand(4,4))
B = tensor(np.random.rand(2,3,4))
b = tensor(np.random.rand(3,4))
x = tensor(np.random.rand(4))
y = tensor(np.random.rand(3))

def test(a,b, f=np.array_equal):
    assert f(a,b)

close = lambda a,b:np.allclose(a,b,1e-10,1e-10)

test(A.ii._, np.trace(A))
test(+A.ii, np.trace(A))

test(A.ii.i, np.diagonal(A))

# use einsum rather than np.sum because results *do* differ by 1e-15
test(A.ij.i, np.einsum('ij->i',A))
test(A.ij.j, np.einsum('ij->j',A))
test(A.ij._, np.einsum('ij->',A))

test(A.ij.ji, np.transpose(A))
test(B.ijk.jki, np.moveaxis(B, (0,1,2), (2,0,1))) # moveaxis is strange

test(+(x.i*x.i), np.dot(x,x), close)
test((x.i*y.j).ij, np.outer(x,y), close)

test(+(A.ij*x.j), A@x)
test((A.ij*x.j).i, A@x)

test((B.ijk*b.jk).i, np.tensordot(B, b, 2), close)

test((A.ij*x.j).ij, A*x, close)
test((A.ij*x.i).ij, A*x.reshape((4,1)), close)


# ## Further work
# 
# integrate einops
# 
# * `A.kij.to('k(ij)')` would vectorize last 2 axes of A (rearrange)
# * `A.ij.max('j')` would reduce along axis i taking the max
# 
# do tensor convolution e.g. `convolve(A.ixy, filt.xy)` or `A.ixy%filt.xy`

# In[ ]:




