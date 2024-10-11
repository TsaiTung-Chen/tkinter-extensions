#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 15:18:35 2024

@author: tungchentsai
"""

idc_skip = list(range(2, 6+1))
i1 = 1
i2 = 10


indices = list(range(i1, i2+1))
print(indices)

counts = 0
for i, idx in enumerate(indices):
    counts += 1
    skip = idx in idc_skip
    print(idx, skip)
    if skip:
        continue

print('Counts:', counts)

