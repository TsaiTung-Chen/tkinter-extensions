#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 23:35:14 2025

@author: tungchentsai
"""

import ttkbootstrap as  ttk

from tkinter_extensions.dialogs import Querybox


root = ttk.Window()
result = Querybox.get_float(
    title='Test',
    prompt='Enter a number.',
    wait=True,
    callback=print
)
root.destroy()

