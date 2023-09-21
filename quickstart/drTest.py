import drjit as dr
import mitsuba as mi
import numpy as np

mi.set_variant("cuda_ad_rgb")
a = dr.full(mi.Float, 0.1, 5)
# the printed result is quite interesting
print(a)
b = dr.linspace(mi.Float, -1, 1, 10)
print(b)
# important: the usage of "select"
"""
A mask (or Bool) is an array of boolean values that can be used to disable arithmetic operations on part of an array. 
It is possible to create such masks with any regular boolean arithmetic (e.g. >, <, >=, <=).

Often time, we combine masks with the dr.select(mask, a, b) statement which correspond to the ternary statement 
mask ? a : b. This is similar to the np.where function in NumPy.
"""
array = dr.arange(mi.Float, 5)
print(array)
m = array > 2.0
print(m)
result = dr.select(m, 2.0, 1.0)
print(result)
