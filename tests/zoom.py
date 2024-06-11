#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 19:20:12 2024

@author: tungchentsai
"""

import tkinter as tk
import tkinter.font

import ttkbootstrap as ttk


root = ttk.Window('Zooming Test', themename='cyborg', size=[800, 600])

ratio = 0.5
def _scale():
    global ratio
    ratio = 1. / ratio
    canvas.scale('all', 1, 1, ratio, ratio)
    font = tk.font.nametofont(canvas.itemcget('text', 'font'))
    new_size = int(font.actual('size') * ratio)
    font.configure(size=new_size)
    canvas.itemconfigure('text', text=f"""font size {font_backup.actual('size')}
bbox all {canvas.bbox('all')}""")

bt = ttk.Button(root, text='scale', command=_scale)
bt.pack(pady=10)

canvas = tk.Canvas(root)
canvas.pack(fill='both', expand=1)

canvas.create_rectangle(1, 1, 300, 200, fill='gray', width=0, outline='cyan')
canvas.create_rectangle(150, 100, 500, 250, fill='magenta', width=1)
font_backup = tk.font.Font()  # keep the reference
canvas.create_text(
    300, 200,
    font=font_backup,
    tags='text'
)
canvas.itemconfigure('text', text=f"""font size {font_backup.actual('size')}
bbox all {canvas.bbox('all')}""")

root.mainloop()

