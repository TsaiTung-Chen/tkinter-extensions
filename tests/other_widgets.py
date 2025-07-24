#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 23:36:40 2025

@author: tungchentsai
"""

import ttkbootstrap as ttk

from tkinter_extensions.widgets import Window, ColorButton


def _keep_changing_color(button: ColorButton, colors: list, ms: int):
    colors = list(colors)
    old_color = button._background
    old_idx = colors.index(old_color)
    new_idx = old_idx + 1
    if new_idx >= len(colors):
        new_idx = 0
    new_color = colors[new_idx]
    button.set_color(new_color)
    button.after(ms, _keep_changing_color, button, colors, ms)

def _new_error():
    def _raise_error():
        raise ValueError("Error Catched")
    
    error_lb.configure(text='No error')
    root.after(1000, _raise_error)
#

root = Window(size=(800, 600))

ttk.Button(root, text='Raise error', command=_new_error).pack(pady=[3, 0])
error_lb = ttk.Label(root, text='No error', bootstyle='inverse')
error_lb.pack(pady=[1, 3])
root._error_message.trace_add(
    'write',
    lambda name, *_: error_lb.configure(text=root.getvar(name))
)

ttk.Button(root, text='Normal Button', bootstyle='danger').pack(pady=[3, 0])

red_bt = ColorButton(root, text='Red Button')
red_bt.pack(pady=[1, 0])
red_bt.set_color('red')

varying_bt = ColorButton(root, text='Varying Button')
varying_bt.pack(pady=[1, 3])
varying_bt.set_color('red')

colors = ['red', 'blue', 'green']
root.after(1500, _keep_changing_color, varying_bt, colors, 1500)
root.after(10000, lambda: root.style.theme_use('cyborg'))

root.mainloop()

