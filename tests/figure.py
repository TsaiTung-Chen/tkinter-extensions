#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 16 23:36:12 2025

@author: tungchentsai
"""

from tkinter_extensions.widgets import Window, Figure

import numpy as np


root = Window(title='Figure Demo', themename='morph')

x = np.arange(0, 10, 1/480, dtype=float)
y = np.sin(2*np.pi*1*x)
#x = np.array([3, 6, 6, 3, 3], dtype=float)
#y = np.array([-0.5, -0.5, 0.5, 0.5, -0.5], dtype=float)
#y = 10**x

fig = Figure(root, toolbar=True)
fig.pack(fill='both', expand=True)

suptitle = fig.set_suptitle('<Suptitle>')
plts = fig.set_plots(1, 1)
plts = np.atleast_2d(plts)
for r in range(plts.shape[0]):
    for c in range(plts.shape[1]):
        plt = plts[r, c]
        for i in range(10):
            plt.line(x, y*i, label=f'line {i} in plot {(r, c)}')
        plt.set_title('<Title>')
        plt.set_tlabel('<top-label>')
        plt.set_blabel('<bottom-label>')
        plt.set_llabel('<left-label>')
        plt.set_rlabel('<right-label>')
        plt.set_ttickslabels(True)
        plt.set_rtickslabels(True)
        #plt.set_lscale('log')
        #plt.set_llimits(10, 150)
        plt.set_llimits(-1.7, 1.7)
        plt.set_legend(True)

root.after(3000, lambda: root.style.theme_use('cyborg'))

root.mainloop()

