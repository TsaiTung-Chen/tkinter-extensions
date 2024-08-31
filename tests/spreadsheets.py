#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 26 17:00:43 2024

@author: tungchentsai
"""

import ttkbootstrap as ttk
import tkinter_extensions as te


root = ttk.Window(title='Book (Root)',
                  themename='morph',
                  position=(100, 100),
                  size=(800, 500))


book = te.widgets.Book(root, scrollbar_bootstyle='round-light')
book.pack(fill='both', expand=True)

book.insert_sheet(1, name='index = 1')
book.insert_sheet(0, name='index = 0')
book.insert_sheet(-1, name='index = -1')

book.after(3000, lambda: root.style.theme_use('minty'))
book.after(4000, lambda: root.style.theme_use('cyborg'))


win = ttk.Toplevel(title='Sheet', position=(100, 100), size=(800, 500))

ss = te.widgets.Sheet(win, scrollbar_bootstyle='light-round')
ss.pack(fill='both', expand=True)

ss.set_foregroundcolors(5, 3, 5, 3, colors='#FF0000', undo=True)
ss.set_backgroundcolors(5, 3, 5, 3, colors='#2A7AD5', undo=True)
ss.resize_cells(5, axis=0, sizes=[80], trace=None, undo=True)

def _set_value_method1():
    ss.set_values(4, 3, 4, 3, values='r4, c3 (method 1)')

def _set_value_method2():
    ss.values.iat[5, 3] = 'R5, C3 (method 2)'
    ss.draw_cells(5, 3, 5, 3)

ss.after(1000, _set_value_method1)
ss.after(2000, _set_value_method2)


root.mainloop()