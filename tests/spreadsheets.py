#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 23:42:10 2025

@author: tungchentsai
"""

import ttkbootstrap as ttk

from tkinter_extensions.widgets import Window, Book, Sheet


root = Window(
    title='Book (Root)',
    themename='morph',
    position=(100, 100),
    size=(800, 500)
)


book = Book(root, scrollbar_bootstyle='round-light')
book.pack(fill='both', expand=True)
book.show_sidebar()  # show the sidebar after `self.pack`

book.insert_sheet(1, name='index = 1')
book.insert_sheet(0, name='index = 0')
book.insert_sheet(-1, name='index = -1')

book.after(4000, lambda: root.style.theme_use('cyborg'))


win = ttk.Toplevel(title='Sheet', position=(100, 100), size=(800, 500))

sh = Sheet(win, shape=(12, 16), scrollbar_bootstyle='light-round')
sh.pack(fill='both', expand=True)

sh.set_foregroundcolors(5, 3, 5, 3, colors='#FF0000', undo=True)
sh.set_backgroundcolors(5, 3, 5, 3, colors='#2A7AD5', undo=True)
sh.resize_cells(5, axis=0, sizes=[80], trace=None, undo=True)

def _set_value_method1():
    sh.set_values(4, 3, 4, 3, data='r4, c3 (method 1)')

def _set_value_method2():
    sh.values[5, 3] = 'R5, C3 (method 2)'
    sh.draw_cells(5, 3, 5, 3)  # update the GUI

sh.after(1000, _set_value_method1)
sh.after(2000, _set_value_method2)

root.mainloop()

