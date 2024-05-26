#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 26 16:56:57 2024

@author: tungchentsai
"""

import numpy as np
import ttkbootstrap as ttk
import tkinter_extensions as te
import matplotlib.pyplot as plt


root = ttk.Window(
    title='Embedding in Ttk', themename='cyborg', size=[500, 500])

t = np.arange(0, 3, .01)
x = 2 * np.sin(2 * np.pi * 1 * t)
fig = plt.Figure(figsize=(5, 4), dpi=100)
ax = fig.subplots()
line, = ax.plot(t, x, label='f = 1 Hz')
ax.set_xlabel("time [s]")
ax.set_ylabel("f(t)")
ax.legend(loc='upper right')

plotter = te.widgets.BasePlotter(root, fig)
plotter.pack(side='top', fill='both', expand=1)

def _update_frequency(new_val):
    f = float(new_val)
    
    # update data
    y = 2 * np.sin(2 * np.pi * f * t)
    line.set_ydata(y)
    line.set_label(f'f = {f: .2f} Hz')
    ax.legend(loc='upper right')
    plotter.figcanvas.draw_idle()  # update canvas

slider = ttk.Scale(root,
                   from_=1,
                   to=5,
                   orient='horizontal',
                   command=_update_frequency)
slider.pack(side='bottom', pady=10)

root.mainloop()

