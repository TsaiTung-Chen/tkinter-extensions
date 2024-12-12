#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 02:52:27 2024

@author: tungchentsai
"""

import tkinter as tk
import numpy as np


arr = np.array(
    [['1', '23', '4'], ['5', '67', '8901']],
    dtype=np.dtypes.StringDType
)

table = '\n'.join( '\t'.join(row) for row in arr )

root = tk.Tk()
root.wm_withdraw()

root.clipboard_clear()
root.clipboard_append(table)

root.update()
root.destroy()

