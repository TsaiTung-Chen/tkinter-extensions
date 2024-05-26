#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 26 16:53:03 2024

@author: tungchentsai
"""

import ttkbootstrap as ttk
import tkinter_extensions as te


root = ttk.Window(themename='cyborg')

for orient, labelanchor, sep in [('vertical', 'n', ' '),
                                 ('horizontal', 'w', '\n')]:
    text = sep.join(['Click', 'me', 'to', 'collapse'])
    collapsed = te.widgets.CollapsedFrame(root,
                                          text=text,
                                          orient=orient,
                                          labelanchor=labelanchor,
                                          bootstyle='success-round-toggle')
    collapsed.grid(sticky='w', padx=6, pady=6)
    ttk.Label(collapsed, text='Label 1').pack()
    ttk.Label(collapsed, text='Label 2').pack()

root.mainloop()

