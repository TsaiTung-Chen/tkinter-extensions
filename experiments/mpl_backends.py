#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 02:15:12 2024

@author: tungchentsai
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


mpl.use('tkagg')

t = np.arange(0., 10, 1/48000)
phi = 2 * np.pi * np.arange(0., 1, 0.25)
x = np.sin(2*np.pi*1*t[:, None] + phi[None , :])

plt.plot(x)
plt.show()

