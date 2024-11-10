#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 18:46:24 2024

@author: tungchentsai
"""

import numpy as np
import ttkbootstrap as ttk
import matplotlib.pyplot as plt

import tkinter_extensions.widgets as wid


root = ttk.Window(title='Set sash position', themename='morph', size=(1000, 600))

nb = ttk.Notebook(root)
nb.pack(fill='both', expand=True)

t = np.linspace(0., 10., 10*48000, endpoint=False)
x = np.sin(2.*np.pi*3.*t)
fig = plt.Figure(figsize=(9, 6), dpi=80, layout='constrained')
fig.suptitle('Sine wave', size=15)
ax = fig.subplots()
ax.plot(t, x)

for i in range(2):
    tab_pw = ttk.Panedwindow(nb, orient='horizontal')
    nb.add(tab_pw, text=f'Tab {i}')
    nb.select(i)
    
    left = wid.ScrolledFrame(tab_pw)
    right = wid.BasePlotter(tab_pw, fig)
    
    ttk.Button(left, text='button').pack(fill='x', expand=True)
    
    tab_pw.add(left.container)
    tab_pw.add(right)
    
    # Set the sash position
    # Note that `panedwindow.sashpos` does not work before the panedwindow and
    # its descendents get a layout manager. Additionally,
    # `widget.update_idletasks` must be called first.
    tab_pw.update_idletasks()  # update all widgets
    print(tab_pw.sashpos(0, 500))  # set the sash position

root.mainloop()

