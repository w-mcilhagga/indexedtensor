import numpy as np
from collections import Counter


def fixname(name):
    """interprets dunder and underscores in the index name"""
    # replace __ with ellipses
    name = name.replace("__", "...")
    if name == "_":
        name = ""
    return name


def op_or_indexed(obj):
    """True if obj is an Einop object or an indexed Tensor"""
    return isinstance(obj, Einop) or (
        type(obj) is Tensor and obj.indices is not None
    )


def implicit_indices(idx):
    """the default summation for indices given in the string idx"""
    idx = Counter(idx)
    defidx = ""
    for e, c in idx.items():
        if c == 1:
            defidx += e
    return "".join(sorted(defidx))


class Tensor(np.ndarray):
    """Tensor object is a numpy array with added einstein indices property and attribute access"""

    
    def __new__(
        subtype,
        shape,
        dtype=float,
        buffer=None,
        offset=0,
        strides=None,
        order=None,
        info=None,
    ):
        t = np.ndarray.__new__(
            subtype, shape, dtype, buffer, offset, strides, order
        )
        t.indices = None  # unindexed
        return t

    def __array_finalize__(self, obj):
        """needed when subclassing ndarray, adds the indices attribute"""
        if obj is not None:
            self.indices = getattr(obj, "indices", None)

    def __getattr__(self, name):
        """define indices for einstein summation/addition.
        The indices are letters (i, j, k, . . .) or _ or __.
        The double underscore behaves like ellipses
        The single underscore is used to summate over all indices.
        This returns a view of the original tensor.
        """
        name = fixname(name)
        if self.indices is None:
            # we are defining the indices of the tensor
            if "..." not in name and len(name) != self.ndim:
                # TODO: what about '...ijk' in a 2d tensor?
                raise ValueError("wrong number of indices")
            s = self.view()
            s.indices = name
            return s
        else:
            # we have already defined the indices of the tensor and now we are
            # summing, e.g. x.ij.i sums over j
            #               x.ij._ sums over i and j
            if name == self.indices:
                self.indices = None
                return self
            else:
                t = tensor(
                    np.einsum(self.indices + "->" + fixname(name), self)
                )
                t.indices = name
                return t
            
    def index(self, idx=None):
        '''applies the indexing to self rather than view'''
        self.indices = None
        if idx is not None:
            self.indices = getattr(self, idx).indices
        return self

    def __pos__(self):
        """default evaluation"""
        if self.indices is None:
            return self
        else:
            return getattr(self, implicit_indices(self.indices))

    def __add__(self, other):
        """set up an einstein-notation broadcast"""
        if self.indices is None:
            if not op_or_indexed(other):
                return np.add(self, other)
        else:
            if not isinstance(other, (Tensor, Einop)):
                return np.add(self, other)
            elif op_or_indexed(other):
                return Einadd(self, other)
        # no return occurred, so invalid
        raise RuntimeError("both indices must be defined in a Tensor addition")

    def __radd__(self, other):
        # commutative
        return self.__add__(other)

    def __mul__(self, other):
        """set up an einstein summation."""
        if self.indices is None:
            if not op_or_indexed(other):
                return np.multiply(self, other)
        else:
            if not isinstance(other, (Tensor, Einop)):
                return np.multiply(self, other)
            elif op_or_indexed(other):
                return Einsum(self, other)
        # no return occurred, so invalid
        raise RuntimeError(
            "both indices must be defined in a Tensor multiplication"
        )

    def __rmul__(self, other):
        # this is commutative
        return self.__mul__(other)
    
    def to(self, pattern, *args):
        '''implements einops with slightly different indexing syntax'''
        if self.indices is None:
            raise RuntimeError('Tensor must be indexed to call to() method')
        pass


import re

def parse_pattern(pattern):
    '''a pattern is a sequence of
    letter + one of /, *, % + number + optional space
    letter + optional space
    number + optional space'''
    pattre = r'([A-Za-z](?:\/|\*|\%)[0-9]+)|([A-Za-z])|([0-9]+)|(\((?: *[A-Za-z] *)+\))|(.)'
    return re.findall(pattre, pattern)

# creation routines


def tensor(object, *args, **kwargs):
    """create an unindexed tensor like np.array()"""
    return np.asarray(object, *args, **kwargs).view(Tensor)


def zeros(shape):
    return tensor(np.zeros(shape))


def ones(shape):
    return tensor(np.ones(shape))


## operations. We need to add numpy functions to this.


class Einop:
    """an object holding an unevaluated einsum/einadd"""

    def __init__(self, *operands):
        self.operands = operands

    def __add__(self, other):
        """add an einsum object to a tensor or einsum"""
        if type(self) == Einadd:
            if type(other) == Einadd:
                return Einadd(*self.operands, *other.operands)
            if op_or_indexed(other):
                return Einadd(*self.operands, other)
        if type(self) == Einsum and op_or_indexed(other):
            return Einadd(self, other)
        raise ValueError("can only operate with indexed Tensors or Einops")

    def __radd__(self, other):
        """multiply an einsum object by a tensor or einsum"""
        return self.__add__(other)

    def __mul__(self, other):
        """multiply an einsum object by a tensor or einsum"""
        if type(self) == Einsum:
            if type(other) == Einsum:
                return Einsum(*self.operands, *other.operands)
            if op_or_indexed(other):
                return Einsum(*self.operands, other)
        if type(self) == Einadd and op_or_indexed(other):
            return Einsum(self, other)
        raise ValueError("can only operate with indexed Tensors or Einops")

    def __rmul__(self, other):
        """multiply an einsum object by a tensor or einsum"""
        return self.__mul__(other)


class Einsum(Einop):
    """an object holding an unevaluated einsum"""

    @property
    def indices(self):
        """the contraction indices"""
        allindices = "".join(map(lambda o: o.indices, self.operands))
        return implicit_indices(allindices)

    def __pos__(self):
        """sum evaluation"""
        return getattr(self, self.indices)

    def __getattr__(self, name):
        """evaluates the product"""
        script = []
        operands = []
        for op in self.operands:
            script.append(op.indices)
            if type(op) == Tensor:
                # put it in the oplist
                operands.append(op)
            else:
                # evaluate and put it in the oplist (op is an Einadd)
                operands.append(getattr(op, op.indices))
        name = fixname(name)
        t = tensor(
            np.einsum(",".join(script) + "->" + name, *operands)
        )
        t.indices = name
        return t


class Einadd(Einop):
    """an object holding an unevaluated broadcast summation"""

    @property
    def indices(self):
        """return all the unique indices"""
        return "".join(
            sorted(set("".join(map(lambda o: o.indices, self.operands))))
        )

    def __pos__(self):
        """add evaluation, no contraction occurs"""
        return getattr(self, self.indices)

    def __getattr__(self, name):
        # index each operand by its own indices
        operands = []
        shape = {}
        for idx in self.indices:
            shape[idx] = 0
        for op in self.operands:
            idx = op.indices
            op = tensor(getattr(op, idx))
            # need this if op is just a tensor
            op.indices = idx
            operands.append(op)
            opshape = dict(zip(idx, op.shape))
            for i in opshape:
                shape[i] = max(shape[i], opshape[i])
        sum = 0.0
        for op in operands:
            b = broadcast_tensor(op, shape)
            sum += b
        name = fixname(name)
        t = tensor(
            np.einsum("".join(shape.keys()) + "->" + name, sum)
        )
        t.indices = name
        return t


def broadcast_tensor(a, shapedict):
    # broadcasts a over b
    # permute axes of a to suit ordering in b
    reorder = ""
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
