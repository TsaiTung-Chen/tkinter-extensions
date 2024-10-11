#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 15:18:35 2024

@author: tungchentsai
"""

from typing import Generator


idc_skip = list(range(2, 6+1))
i1 = 1
i2 = 10

def table_index_generator(start: int, last: int) -> Generator[int, bool, None]:
    assert isinstance(start, int), (type(start), start)
    assert isinstance(last, int), (type(last), last)
    assert 0 <= start <= last, (start, last)
    
    n_half = (start + last) // 2
    upper_idc = list(range(start, n_half+1, +1))  # 1st half (upper)
    lower_idc = list(range(last, n_half, -1))  # 2nd half (lower)
    indices = [None] * (len(upper_idc) + len(lower_idc))
    indices[0::2] = upper_idc
    indices[1::2] = lower_idc
    print(indices)
    
    start_skip: bool = False
    for i, idx in enumerate(indices):
        start_skip = yield idx
        if start_skip:
            break
    else:
        return
    
    # Start to skip from current `idx` => now find the end index where skip ends
    if i % 2 == 0:  # current `idx` is in `upper_idc`
        indices = lower_idc[i//2:] + upper_idc[i//2+1:][::-1]
    else:  # current `idx` is in `lower_idc`
        indices = upper_idc[(i+1)//2:] + lower_idc[(i+1)//2+1:][::-1]
    
    for idx in indices:
        end_skip = yield idx
        if end_skip:
            break


index_generator = table_index_generator(i1, i2)
counts = 1
try:
    idx = next(index_generator)
    while True:
        skip = idx in idc_skip
        print(idx, skip)
        idx = index_generator.send(skip)
        counts += 1
except StopIteration:
    pass
print('Counts:', counts)

