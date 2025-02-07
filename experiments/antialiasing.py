#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 21:03:49 2025

@author: tungchentsai
"""

import tkinter as tk
import numpy as np


def drop_consecutive_duplicates(points):
    points = np.asarray(points)  # [[x1, y1], [x2, y2], ...]
    assert points.ndim == 2, points.shape    
    
    if points.size <= 2:
        return points
    
    retain = np.diff(points[:-1], axis=0).any(axis=1)
    
    return np.concat(
        (points[:1], points[1:-1][retain], points[-1:]),
        axis=0
    )


WIDTH = 5
WIDTH_INC = 1

root = tk.Tk()

canvas = tk.Canvas(root, bg='white', width=600, height=600)
canvas.pack(fill='both', expand=True)

# No antialiasing
x = np.linspace(50, 80, 1000, endpoint=False)
y = np.linspace(50, 300, 1000, endpoint=False)
points = np.concat((x[:, None], y[:, None]), axis=1)
points = drop_consecutive_duplicates(np.floor(points))
canvas.create_line(*points.ravel(), fill='#000', width=WIDTH)

# Antialiasing (approx 33% of color intensity)
points2 = points.copy()
points2[:, 0] += 50
canvas.create_line(*points2.ravel(), fill='#AAA', width=WIDTH + WIDTH_INC)
canvas.create_line(*points2.ravel(), fill='#000', width=WIDTH)

root.mainloop()

