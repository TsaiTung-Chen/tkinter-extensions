#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 26 16:55:17 2024

@author: tungchentsai
"""

import random
import ttkbootstrap as ttk
import tkinter_extensions as te

root = ttk.Window(title='Drag and Drop', themename='cyborg')

container = te.widgets.OrderlyContainer(root)
container.pack(fill='both', expand=1)
buttons = list()
for r in range(6):
    buttons.append(list())
    for c in range(3):
        dash = '----' * random.randint(1, 5)
        button = ttk.Button(container,
                            text=f'|<{dash} ({r}, {c}) {dash}>|',
                            takefocus=True,
                            bootstyle='outline')
        buttons[-1].append(button)
container.dnd_put(buttons,
                  sticky='nse',
                  expand=True,
                  padding=10,
                  ipadding=6)

window = ttk.Toplevel(title='Button Trigger Drag and Drop', topmost=True)
window.lift()
window.after(300, window.focus_set)
container = te.widgets.TriggerOrderlyContainer(window)
container.pack(fill='both', expand=1)
frames = list()
for r in range(4):
    frames.append(list())
    for c in range(3):
        dash = '----' * random.randint(1, 5)
        frame = ttk.Frame(container)
        trigger = ttk.Button(frame,
                             text='::',
                             takefocus=True,
                             cursor='hand2',
                             bootstyle='success-link')
        trigger.pack(side='left')
        trigger._dnd_trigger = True
        ttk.Label(frame,
                  text=f'|<{dash} ({r}, {c}) {dash}>|',
                  bootstyle='success').pack(side='left')
        frames[-1].append(frame)
container.dnd_put(frames,
                  sticky='nsw',
                  expand=(False, True),
                  padding=10,
                  ipadding=6)
window.place_window_center()

root.place_window_center()
root.mainloop()

