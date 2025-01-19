#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 21:03:49 2025

@author: tungchentsai
"""

import time
import tkinter as tk
import numpy as np


def drop_consecutive_duplicates(points):
    points = np.asarray(points)  # [[x1, y1], [x2, y2], ...]
    assert points.ndim == 2, points.shape    
    
    if points.size == 0:
        return points
    
    retain = np.diff(points, axis=0).any(axis=1)
    
    return np.concat((points[:1], points[1:][retain]), axis=0)


def rdp(points, epsilon=0.8):  # method 1 (recursive method)
    """
    Simplifies a series of points using the RDP algorithm.
    """
    if points.shape[0] < 3:
        return points
    
    # Find the point with the maximum distance
    start, between, end = points[0], points[1:-1], points[-1]
    x1, y1 = start
    x2, y2 = end
    xs, ys = between[:, 0], between[:, 1]
    if (start == end).all():
        dists = np.sqrt((xs - x1)**2 + (ys - y1)**2)
    else:
        num = np.abs((y2 - y1)*xs - (x2 - x1)*ys + x2*y1 - y2*x1)
        den = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
        dists = num / den
    index = np.argmax(dists)
    max_dist = dists[index]
    index += 1
    
    if max_dist < epsilon:  # distances are not large enough
        return np.asarray([start, end])  # => linear
    
    # Recursively simplify the curve
    first_half = rdp(points[:index + 1], epsilon)
    second_half = rdp(points[index:], epsilon)
    
    return np.concat((first_half[:-1], second_half), axis=0)


def ppd(points, ratio=0.001):  # method 2 (preferred)
    """
    Simplifies a series of points based on the distance ratios.
    """
    if points.shape[0] < 3:
        return points
    
    # Get points
    x1s, y1s = points[:-2, 0], points[:-2, 1]  # front points
    xs, ys = points[1:-1, 0], points[1:-1, 1]  # middle points
    x2s, y2s = points[2:, 0], points[2:, 1]  # back points
    
    # Calculate the perpendicular distances from middle points to front-back lines
    numerator = np.abs((y2s - y1s)*xs - (x2s - x1s)*ys + x2s*y1s - y2s*x1s)
    denominator = base_dists = np.sqrt((x2s - x1s)**2 + (y2s - y1s)**2)
    with np.errstate(divide='ignore', invalid='ignore'):
        dists = numerator / denominator
        ratios = dists / (base_dists/2)
    ratios[np.isnan(ratios)] = 0
    retain = ratios >= ratio
    
    return np.concat((points[:1], points[1:-1][retain], points[-1:]), axis=0)


def draw(points):
    canvas.delete('all')
    t0 = time.monotonic()
    points = drop_consecutive_duplicates(points)
    t1 = time.monotonic()
    points = ppd(points)
    t2 = time.monotonic()
    points = points.astype(int)
    canvas.create_line(*points.ravel(), fill='red', width=1)
    t3 = time.monotonic()
    print(f'Drop consecutive duplicates: {t1 - t0: .6f} s')
    print(f'Simplify line: {t2 - t1: .6f} s')
    print(f'Create line: {t3-t2: .6f} s')
    print(f'Total time: {t3 - t0: .6f} s')


root = tk.Tk()
root.wm_geometry('800x600')

canvas = tk.Canvas(root, bg='white')
canvas.pack(fill='both', expand=True)
canvas.bind('<Configure>', lambda e: draw(dense_points))

x = 100 * np.arange(0, 10, 1/48000, dtype=float)
y = 100 * (np.sin(2*np.pi*1/200*x) + 2)
dense_points = np.concat((x[:, None], y[:, None]), axis=1).round()

root.mainloop()

