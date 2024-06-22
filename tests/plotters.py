#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 26 16:56:57 2024

@author: tungchentsai
"""

import numpy as np
import ttkbootstrap as ttk
import tkinter_extensions as te
from tkinter_extensions.widgets.plotters import RC
import matplotlib.pyplot as plt


root = ttk.Window(
    title='Embedding in Ttk', themename='cyborg', size=[600, 600])

t = np.arange(0, 3, .01)
x = 2 * np.sin(2 * np.pi * 1 * t)

plt.rcParams.update(RC["dark"])
fig = plt.Figure(figsize=(5, 4), dpi=100)
fig.suptitle('Sine wave')
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

# Switch theme
def _change_to_light_theme():
    style = root.style
    if style.theme_use() == 'morph':
        new_rc = RC["dark"]
        new_theme = 'cyborg'
    else:
        new_rc = RC["light"]
        new_theme = 'morph'
    plt.rcParams.update(new_rc)
    style.theme_use(new_theme)

bt = ttk.Button(root, text='Switch theme', command=_change_to_light_theme)
bt.pack(side='bottom', pady=10)

root.mainloop()

