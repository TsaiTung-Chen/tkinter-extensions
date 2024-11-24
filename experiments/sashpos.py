#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 18:46:24 2024

@author: tungchentsai
"""

import tkinter as tk

import numpy as np
import ttkbootstrap as ttk
import matplotlib.pyplot as plt

import tkinter_extensions.widgets as wid


def sashpos(pw: ttk.Panedwindow, pane: tk.BaseWidget, width: int) -> int:
    """
    Set the sash position
    Note that `pw.sashpos` does not work before the panedwindow and its
    descendents get a layout manager. Additionally, `widget.update_idletasks`
    must be called first.
    """
    pane.update_idletasks()  # update all widgets
    resultant_width = pw.sashpos(0, width)  # set the sash position
    pane.update_idletasks()  # update all widgets
    
    return resultant_width


def plot(title: str) -> plt.Figure:
    t = np.linspace(0., 20., 20*48000, endpoint=False)
    x = np.sin(2.*np.pi*0.2*t)
    
    fig = plt.Figure(figsize=(9, 3), dpi=100, layout='constrained')
    fig.suptitle(title, size=15)
    ax = fig.subplots()
    ax.plot(t, x)
    
    return fig
#

root = ttk.Window(title='Set sash position', size=(900, 600))

nb = ttk.Notebook(root)
nb.pack(fill='both', expand=True)

for i in range(2):
    # Notebook tab
    pw = ttk.Panedwindow(nb, orient='horizontal')
    nb.add(pw, text=f'Tab {i}')
    nb.select(i)
    
    # Left pane
    left_pane = wid.ScrolledFrame(pw, propagate_geometry=False)
    pw.add(left_pane.container)
    
    ttk.Button(left_pane, text='button left').pack(fill='x', expand=True)
    
    # Right pane
    subpw = ttk.Panedwindow(pw, orient='horizontal')
    pw.add(subpw)
    
    # Left subpane in the right pane
    left_subpane = wid.ScrolledFrame(subpw, propagate_geometry=False)
    subpw.add(left_subpane.container)
    
    ttk.Button(left_subpane, text='button middle').pack(fill='x', expand=True)
    
    # Right subpane in the right pane
    right_subpane = wid.ScrolledFrame(subpw)
    subpw.add(right_subpane.container)
    
    for j in range(5):
        wid.BasePlotter(
            right_subpane, plot(f'Sine Wave {i*5 + j}')
        ).pack(fill='x', expand=True, pady=(0, 18))
    
    # Set the positions of the sashes
    print(sashpos(pw, left_pane, width=300))
    print(sashpos(subpw, left_subpane, width=300))

root.mainloop()

