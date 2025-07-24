#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 23:33:05 2025

@author: tungchentsai
"""

import random
import tkinter as tk

import ttkbootstrap as ttk

from tkinter_extensions.widgets import (
    Window, DnDContainer, OrderlyDnDItem, RearrangedDnDContainer
)


root = Window(title='Drag and Drop', themename='cyborg')

container = DnDContainer(root)
container.pack(side='bottom', fill='both', expand=1)
items = []
for r in range(6):
    items.append([])
    for c in range(3):
        dash = '----' * random.randint(1, 5)
        item = OrderlyDnDItem(container, selectbutton=True)
        ttk.Button(
            item,
            text=f'|<{dash} ({r}, {c}) {dash}>|',
            takefocus=False,
            bootstyle='outline'
        ).pack(fill='both', expand=True)
        items[-1].append(item)
container.dnd_put(
    items,
    sticky='nse',
    expand=True,
    padding=10,
    ipadding=6
)
var = tk.BooleanVar(root, value=False)
rearrange_bt = ttk.Checkbutton(
    root,
    text='Rearrange',
    variable=var,
    command=container.toggle_rearrangement
)
rearrange_bt.pack(side='top')
root.place_window_center()
root.update_idletasks()

top = ttk.Toplevel(title='Button Trigger Drag and Drop')
top.lift()
top.after(300, top.focus_set)
container = RearrangedDnDContainer(top)
container.pack(side='bottom', fill='both', expand=True)
items = list()
for r in range(4):
    items.append([])
    for c in range(3):
        dash = '----' * random.randint(1, 5)
        item = OrderlyDnDItem(container, selectbutton=True, dragbutton=True)
        ttk.Label(
            item,
            text=f'|<{dash} ({r}, {c}) {dash}>|',
            bootstyle='success'
        ).pack(fill='both', expand=True)
        items[-1].append(item)
container.dnd_put(
    items,
    sticky='nsw',
    expand=(False, True),
    padding=(3, 6),
    ipadding=12
)
top.place_window_center()

root.mainloop()

