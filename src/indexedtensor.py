import torch as tc
from collections import Counter


def op_or_indexed(obj):
    """True if obj is an Einop object or an indexed ITensor"""
    return isinstance(obj, Einop) or (
        type(obj) is ITensor and obj._indices_ is not None
    )


def implicit_indices(idx):
    """the default summation for indices given in the string idx"""
    idx = Counter(idx)
    defidx = ""
    for e, c in idx.items():
        if c == 1:
            defidx += e
    return "".join(sorted(defidx))

pointops = [tc.sin, tc.cos, tc.abs]

class ITensor(tc.Tensor):
    """Tensor object is a torch tensor with added einstein indices property and attribute access"""

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        '''passes through all torch functions; we should keep indices intact'''
        if kwargs is None:
            kwargs = {}
        result = super().__torch_function__(func, types, args, kwargs)
        if type(result) == ITensor:
            # for point ops, the indices should pass through
            if func in pointops:
                result.reindex(args[0]._indices_)
            else:
                result.reindex()
        return result

    def __new__(subtype, *args, **kwargs):
        t = tc.Tensor.__new__(subtype, *args, **kwargs)
        t._indices_ = None  # unindexed
        return t
    
    def __getattr__(self, name):
        """define indices for einstein summation/addition.
        The indices are letters (i, j, k, . . .) or ... .
        This returns a view of the original tensor.
        """
        if name[0]=='_' and name!='_' and name!='__':
            # torch internals sometimes use getattr(tensor, name, None) 
            # to detect features, we have to trap this.
            raise AttributeError('Indexed Tensor attributes must not start with _ (except _ and __)')
        if name=='_':
            name=''
        return self[name]
    
    def __getitem__(self, name):
        """define indices for einstein summation/addition.
        The indices are letters (i, j, k, . . .) or ... .
        This returns a view of the original tensor.
        """
        if type(name) != str:
            return super().__getitem__(name)
        if self._indices_ is None:
            # we are defining the indices of the tensor
            if "->" in name:
                name = name.split("->")
                return self[name[0]][name[1]]
            if "..." not in name and len(name) != self.ndim:
                # TODO: what about '...ijk' in a 2d tensor?
                raise ValueError("wrong number of indices [" + name + "]")
            s = self.view(self.shape)
            s._indices_ = name
            return s
        else:
            # we have already defined the indices of the tensor and now we are
            # summing, e.g. x['ij']['i'] sums over j
            #               x['ij'][''] sums over i and j
            if name == self._indices_:
                self._indices_ = None
                return self
            else:
                t = ITensor(tc.einsum(self._indices_ + "->" + name, self))
                t._indices_ = name
                return t

    def reindex(self, idx=None):
        """applies the indexing to self rather than view"""
        self._indices_ = None
        if idx is not None:
            self._indices_ = self[idx]._indices_
        return self

    def __pos__(self):
        """default evaluation"""
        if self._indices_ is None:
            return self
        else:
            return self[implicit_indices(self._indices_)]

    def __add__(self, other):
        """set up an einstein-notation broadcast"""
        if self._indices_ is None:
            if not op_or_indexed(other):
                return tc.add(self, other)
        else:
            if not isinstance(other, (ITensor, Einop)):
                return tc.add(self, other)
            elif op_or_indexed(other):
                return Einadd(self, other)
        # no return occurred, so invalid
        raise RuntimeError(
            "both indices must be defined in a ITensor addition"
        )

    def __radd__(self, other):
        # commutative
        return self.__add__(other)

    def __mul__(self, other):
        """set up an einstein summation."""
        if self._indices_ is None:
            if not op_or_indexed(other):
                return tc.multiply(self, other)
        else:
            if not isinstance(other, (ITensor, Einop)):
                return tc.multiply(self, other)
            elif op_or_indexed(other):
                return Einsum(self, other)
        # no return occurred, so invalid
        raise RuntimeError(
            "both indices must be defined in a ITensor multiplication"
        )

    def __rmul__(self, other):
        # this is commutative
        return self.__mul__(other)

    def to(self, pattern, *args):
        """implements einops with slightly different indexing syntax"""
        if self._indices_ is None:
            raise RuntimeError("ITensor must be indexed to call to() method")
        pass


import re


def parse_pattern(pattern):
    """a pattern is a sequence of
    letter + one of /, *, % + number + optional space
    letter + optional space
    number + optional space"""
    pattre = r"([A-Za-z](?:\/|\*|\%)[0-9]+)|([A-Za-z])|([0-9]+)|(\((?: *[A-Za-z] *)+\))|(.)"
    return re.findall(pattre, pattern)


# creation routines


def tensor(object, *args, **kwargs):
    """create an unindexed tensor like np.array()"""
    if isinstance(object, tc.Tensor):
        return ITensor(object)
    else:
        return ITensor(tc.tensor(object, *args, **kwargs))


def zeros(shape):
    return ITensor(tc.zeros(shape))


def ones(shape):
    return ITensor(tc.ones(shape))


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
        raise ValueError("can only operate with indexed ITensors or Einops")

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
        raise ValueError("can only operate with indexed ITensors or Einops")

    def __rmul__(self, other):
        """multiply an einsum object by a tensor or einsum"""
        return self.__mul__(other)


class Einsum(Einop):
    """an object holding an unevaluated einsum"""

    @property
    def _indices_(self):
        """the contraction indices"""
        allindices = "".join(map(lambda o: o._indices_, self.operands))
        return implicit_indices(allindices)

    def __pos__(self):
        """sum evaluation"""
        return self[self._indices_]

    
    def __getattr__(self, name):
        if name[0]=='_' and name!='_':
            raise AttributeError('nope')
        if name=='_':
            name=''
        return self[name]
    
    def __getitem__(self, name):
        """evaluates the product"""
        script = []
        operands = []
        for op in self.operands:
            script.append(op._indices_)
            if type(op) == ITensor:
                # put it in the oplist
                operands.append(op)
            else:
                # evaluate and put it in the oplist (op is an Einadd)
                operands.append(getattr(op, op._indices_))
        t = ITensor(tc.einsum(",".join(script) + "->" + name, *operands))
        t._indices_ = name
        return t


class Einadd(Einop):
    """an object holding an unevaluated broadcast summation"""

    @property
    def _indices_(self):
        """return all the unique indices"""
        return "".join(
            sorted(set("".join(map(lambda o: o._indices_, self.operands))))
        )

    def __pos__(self):
        """add evaluation, no contraction occurs"""
        return self[self._indices_]

    
    def __getattr__(self, name):
        if name[0]=='_' and name!='_':
            raise AttributeError('nope')
        if name=='_':
            name=''
        return self[name]

    def __getitem__(self, name):
        # index each operand by its own indices
        operands = []
        shape = {}
        for idx in self._indices_:
            shape[idx] = 0
        for op in self.operands:
            idx = op._indices_
            op = tensor(op[idx])
            # need this if op is just a tensor
            op._indices_ = idx
            operands.append(op)
            opshape = dict(zip(idx, op.shape))
            for i in opshape:
                shape[i] = max(shape[i], opshape[i])
        sum = 0.0
        for op in operands:
            b = broadcast_tensor(op, shape)
            sum += b
        t = ITensor(tc.einsum("".join(shape.keys()) + "->" + name, sum))
        t._indices_ = name
        return t


def broadcast_tensor(a, shapedict):
    # broadcasts a over b
    # permute axes of a to suit ordering in b
    reorder = ""
    for i in shapedict.keys():
        if i in a._indices_:
            reorder += i
    a = ITensor(a[reorder])
    a._indices_ = reorder
    # work out new shape of a
    newshape = []
    for i in shapedict.keys():
        try:
            axis = a._indices_.index(i)
            newshape.append(a.shape[axis])
        except:
            newshape.append(1)
    a = tc.reshape(a, newshape)
    a = tc.broadcast_to(a, [*shapedict.values()])
    return a
