# indexed tensor

The `ITensor` class is a subclass of `torch.Tensor` which (ab)uses attribute lookup to define the axes of the tensor.

If `A` is a 3D tensor, then `A.ijk` is an **indexed tensor** which labels the three axes `i`, `j`, and `k` respectively. The number of letters in the
index must match the number of axes in the tensor, except when double-underscores (dunders) are used. `A.i__` with a dunder says the first axis is
labelled `i` and the others have no label, however many there are. `A.__i` says the last axis is labelled `i` and the others have no label, however
many there are, and `A.i__j` labels the first and last axes as `i` and `j`.

An indexed tensor is a view of the original tensor.

Indexed tensors can be multiplied together. For example `A.ij*x.j` represents a tensor whose `[i,j]`-th entry is `A[i,j]*x[j]`. However, this
computation is not carried out until a summation is declared. Summation is indicated by **another** indexing attribute which gives the indices/axes
that are not summed over.

Some examples:

-   `(A.ij*x.j).j` sums the tensor `A.ij*x.j` over the axis `i` (the first axis) to give a 1D tensor $t_j = \sum_i{A_{ij}x_j}$. The brackets
    around `A.ij*x.j` are needed because the attribute operator `.` binds tightly
-   `(A.ij*x.j).i` sums the tensor over the axis `j` (the second axis) to give a 1D tensor $t_i = \sum_j{A_{ij}x_j}$. This is standard matrix-vector multiplication
-   `(A.ij*x.i)._` sums the tensor over the axes `i` and `j` to give $t = \sum_{i,j}{A_{ij}x_j}$. A single underscore index says to sum over all axes.
-   Finally, `(A.ij*x.i).ij` does no summation and yields a 2D tensor with elements $t_{ij} = A_{ij}x_j$

Any time an indexed expression like `A.ij*x.j` is evaluated, it yields an indexed tensor (unless all indices are summed out), which can be indexed again as needed.

The Einstein summation convention is that any repeated indices are summed over. This convention is implemented through the unary plus operator.
For example,<br/> `+(A.ij*x.j)` will sum over the repeated index `j`, and is equivalent to `(A.ij*x.j).i`.

The multiplication and indexing replicates the functionality of the numpy einsum function, because it is implemented using it. The following operations
can be performed with this notation:

-   Trace: `A.ii._` or `+A.ii`
-   Diagonal: `A.ii.i`
-   Summation over one axis: `A.ij.i` or `A.ij.j`, equivalent to `np.sum(A, axis=1)` or `np.sum(A, axis=0)`
-   Summation over all axes: `A.ijk._` is the equivalent of `np.sum(A)`
-   Transpose: `A.ij.ji`
-   Move axis: `A.ijk.jki`
-   Inner (dot) product: `(x.i*x.i)._` or `+(x.i*x.i)`
-   Outer product: `(a.i*b.j).ij`
-   Matrix multiplication: `(A.ij*x.j).i` or `+(A.ij*x.j)`
-   Quadratic form: `(x.i*A.ij*x.j)._` or `(x.i*x.j*A.ij)._` or `+(x.i*A.ij*x.j)` (NB indexed multiplication is commutative)
-   Tensor contractions of all types: `(A.ijk*b.jk).i`, `(A.ijk*b.ik).j`, or `+(A.ijk*b.ik)` etc.
-   Broadcasting: `(A.ij*x.i).ij` over the first axis or `(A.ij*x.j).ij` over the second

`torch.einsum` also lets you use ellipses in summation formulas. In the indexed tensor class, a dunder index `__` is used to represent ellipses.

In all of these cases, the result is an unindexed tensor.

## Indexing.

The attribute access is also implemented through indexing. In particular, `A.ij` is
the same as `A["ij"]`.

## Interoperability

Tensors, either indexed or bare, interact smoothly with torch tensors, functions and scalars. Indexed tensors keep their indices when passed through point operations (e.g.
`sin`, `cos`, `abs`), but lose them otherwise.

For example:

-   `A.ij+5` returns an indexed tensor (with indices `ij`) where every element has been increased by 5. `A+5` would return an unindexed tensor.

-   `A.ij*B` does standard element-by-element multiplication of the indexed tensor `A` and the unindexed tensor or ndarray `B`.

## Tensor Broadcasting

The Tensor class also implements indexed addition, or any mix of indexed addition and multiplication.

Simple cases - no summation

```python
(A.ij+x.i).ij
(A.ij+x.j).ij
```

some summation

```python
(A.ij*b.j+x.i).i # = ((A.ij*b.j).i+x.i).i   no broadcasting occurs
(A.ij*b.j+x.j).i # = ((A.ij*b.j).ij+x.j).i  broadcasting occurs, note that the inner .ij keeps it's indices
```

complicated

```python
((A.ij*b.i+x.i)+y.j).i # = (((A.ij*b.i+x.i).ij+y.j).i
```

# einops

Some einops functions are provided, with a gratuitous change in notation

```python
# reduce
A.ijk.to('ij', np.max)
# downsample, optional spaces are used to disambiguate and for clarity
A.ijk.to('i j/2 k', f)
# same as A.ijk.to('i j/2 k 2').to('ijk', f)
# which is how it should be implemented
# upsample
A.ijk.to('i j*2 k', pattern)
# rearrange by grouping
A.ijk.to('i(jk)')
# ungroup, assuming k is 5
A.ij.to('i j/5 j%5')
# repeat
A.ij.to('ij3')
```

# convolution

```python
convolve(A.ijk, f.k, mode='valid')
A.ijk@f.k # mode defaults to valid
```
