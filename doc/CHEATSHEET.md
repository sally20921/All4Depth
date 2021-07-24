# reshape 
```python
import numpy as np
a1 = np.arange(1,13) # numbers 1 to 12
print(a1.shape)
# (12,)
print(a1)
# [1 2 3 4 5 6 7 8 9 10 11 12]
a1_2d = a1.reshape(3,4) 
print(a1_2d.shape)
# (3,4)
print(a1_2d)
# [[1 2 3 4]
  [5 6 7 8]
  [9 10 11 12]]
```

# reshape along different dimensions
* By default, `reshape()` reshapes the array along the 0th dimension.
`numpy.reshape(a, newshape, order='C')`
* gives a new shape to an array without changing its data
* order: {'C', 'F', 'A'}, optional
* reads the elements of `a` using this order, and place the elements into the reshaped array using this order. 
* `C` means to read/write the elements using C-like order, with the last axis index changing fastest, back to the first axis index changing slowest. 
* `F` means to read/write the elements using Fortran-like index order, with the first index changing the fastest, and last index changing slowest.

# concatenate/stack arrays with `np.stack()` and `np.hstack()`
```python
al = np.arange(1,13)
print(a1)
# [1 2 3 4 5 6 7 8 9 10 11 12]

a2 = np.arange(13, 25)
print(a2)
# [13 14 15 16 17 18 19 20 21 22 23 24]
```
* by default, `np.stack()` stacks arrays along the 0th dimension.
* if `axis=-1` it will be the last dimension
* concatenate as a long 1D array with `np.hstack()` (stack horizontally)
```python
stack_long= np.hstack((a1, a2))
print(stack_long.shape)
# (24,)
print(stack_long)
# [1 2 3 4 ... 23 24]
```

# create multi-dimensional array (3D)
```python
a1 = np.arange(1, 13).reshape(3,-1)
a2 = np.arange(13, 25).reshape(3, -1)
print(a1)
# [[1 2 3 4]
   [5 6 7 8]
   [9 10 11 12]]
print(a2)
# [[13 14 15 16]
    [17 18 19 20]
    [21 22 23 24]]
```

# create a 3D array by stacking the arrays along different axes/dimensions
```python
a3_0 = np.stack((a1, a2)) # default axis=0 (dimension 0)
a3_1 = np.stack((a1, a2), axis=1)
a3_2 = np.stack((a1, a2)), axis=2)

print(a3_0.shape)
# (2,3,4)
print(a3_1.shape)
# (3, 2, 4)
print(a3_2.shape)
# (3, 4, 2)
```

