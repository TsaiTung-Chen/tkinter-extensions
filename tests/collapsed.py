#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 23:37:53 2025

@author: tungchentsai
"""

import ttkbootstrap as ttk

from tkinter_extensions.widgets import Window, CollapsedFrame


root = Window(themename='cyborg')

for orient, labelanchor, sep in [('vertical', 'n', ' '),
                                 ('horizontal', 'w', '\n')]:
    text = sep.join(['Click', 'me', 'to', 'collapse', 'or', 'expand'])
    collapsed = CollapsedFrame(
        root,
        text=text,
        orient=orient,
        labelanchor=labelanchor,
        button_bootstyle='success-round-toggle'
    )
    collapsed.grid(sticky='w', padx=6, pady=6)
    ttk.Label(collapsed, text='Label 1').pack()
    ttk.Label(collapsed, text='Label 2').pack()

root.mainloop()

