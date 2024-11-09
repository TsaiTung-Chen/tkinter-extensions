#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 26 16:56:57 2024

@author: tungchentsai
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

print('Backend:', matplotlib.get_backend())

t = np.arange(0, 3, .01)
x = 2 * np.sin(2 * np.pi * 1 * t)

fig = plt.figure()
fig.suptitle('Sine wave')
ax = fig.subplots()
line, = ax.plot(t, x, label='f = 1 Hz')
ax.set_xlabel("time [s]")
ax.set_ylabel("f(t)")
ax.legend(loc='upper right')

plt.show()

