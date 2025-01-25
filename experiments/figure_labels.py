#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 17:32:37 2025

@author: tungchentsai
"""

import numpy as np

dmin, dmax = (1057, 12106)
max_n = 15
print('(dmin, dmax), max_n:', (dmin, dmax), max_n)

s = np.ceil((dmax - dmin) / (max_n - 2))
exponent, significand = divmod(np.log10(s), 1)
if significand != 0:
    s = 10**exponent * (10**significand).round()

a = np.floor(dmin / s) * s
b = np.ceil(dmax / s) * s
n = (b - a) / s + 1
assert n >= 0 and n % 1 == 0, n
print(f'Step: {s}')
print(f'n: {n}')

print(np.linspace(a, b, int(n), endpoint=True))
