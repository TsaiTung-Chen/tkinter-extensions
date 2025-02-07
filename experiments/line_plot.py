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
    
    if points.size < 3:
        return points
    
    retain = np.diff(points[:-1], axis=0).any(axis=1)
    
    return np.concat(
        (points[:1], points[1:-1][retain], points[-1:]),
        axis=0
    )


def cutoff(points):
    """
    Z patterns usually result from rounding (x, y) coordinate points of a tilt
    line segment. For example,
        a size-1 vertical line in between two horizontal lines:
        ─...─
             │
             ─...─
        
        or a size-1 horizontal line in between two vertical lines:
        │
        .
        .
        .
        │
        ─
         │
         .
         .
         .
         │
    
    We simplify the z patterns by dropping the turning points.
    """
    def _find_z_patterns(dup1, dun1, dvp, dvn):
        du1_idc = []
        for du1 in [dup1, dun1]:
            _du1_idc = du1.nonzero()[0]
            if _du1_idc.size:
                if _du1_idc[0] == 0:
                    _du1_idc = _du1_idc[1:]
                if _du1_idc[-1] == dup1.size - 1:
                    _du1_idc = _du1_idc[:-1]
            du1_idc.append(_du1_idc)
        du1_idc = np.concat(du1_idc)
        
        z_pattern = (dvp[du1_idc - 1] & dvp[du1_idc + 1]) \
                  | (dvn[du1_idc - 1] & dvn[du1_idc + 1])
        
        return du1_idc[z_pattern]
    #> end of _find_z_patterns()
    
    points = np.asarray(points)  # [[x1, y1], [x2, y2], ...]
    assert points.ndim == 2, points.shape    
    
    if points.shape[0] < 4:
        return points
    
    dx, dy = np.diff(points, axis=0).T
    dx0, dy0 = (dx == 0), (dy == 0)
    dxp, dxn = dy0 & (dx > 0), dy0 & (dx < 0)
    dyp, dyn = dx0 & (dy > 0), dx0 & (dy < 0)
    dxp1, dxn1 = (dx == 1), (dx == -1)
    dyp1, dyn1 = (dy == 1), (dy == -1)
    
    z_pattern_x_idc = _find_z_patterns(dxp1, dxn1, dyp, dyn)
    z_pattern_y_idc = _find_z_patterns(dyp1, dyn1, dxp, dxn)
    
    retain = np.ones(points.shape[0] - 1, dtype=bool)
    retain[z_pattern_x_idc - 1] = False  # drop front points
    retain[z_pattern_y_idc] = False  # drop back points
    
    return np.concat((points[:1], points[1:][retain]), axis=0)


def rdp(points, distance=0.707):  # method 1 (recursive method)
    """
    Simplifies a series of points using the Ramer–Douglas–Peucker algorithm.
    
    Ref: https://en.wikipedia.org/wiki/Ramer–Douglas–Peucker_algorithm
    """
    if points.shape[0] < 3:
        return points
    
    # Find the point with the maximum perpendicular distance
    start, between, end = points[0], points[1:-1], points[-1]
    x1, y1 = start
    x2, y2 = end
    xs, ys = between[:, 0], between[:, 1]
    if (start == end).all():
        dists = np.sqrt((xs - x1)**2 + (ys - y1)**2)
    else:
        num = np.abs((y2 - y1)*xs - (x2 - x1)*ys + x2*y1 - y2*x1)
        den = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
        with np.errstate(divide='ignore', invalid='ignore'):  # address 1/0 or 0/0
            dists = num / den
        dists[np.isnan(dists)] = 0.
    index = np.argmax(dists)
    max_dist = dists[index]
    index += 1
    
    if max_dist <= distance:  # distances are not large enough
        return np.asarray([start, end])  # => linear
    
    # Recursively simplify the curve
    first_half = rdp(points[:index + 1], distance=distance)
    second_half = rdp(points[index:], distance=distance)
    
    return np.concat((first_half[:-1], second_half), axis=0)


def rdpr(points, distance=0.707, ratio=0.01):  # method 2 (recursive method)
    """
    Simplifies a series of points using the Ramer–Douglas–Peucker algorithm.
    This modified algorithm also takes the distance ratio into account.
    
    Ref: https://en.wikipedia.org/wiki/Ramer–Douglas–Peucker_algorithm
    """
    if points.shape[0] < 3:
        return points
    
    # Find the point with the maximum perpendicular distance
    start, between, end = points[0], points[1:-1], points[-1]
    x1, y1 = start
    x2, y2 = end
    xs, ys = between[:, 0], between[:, 1]
    base_dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    if base_dist == 0:
        dists = np.sqrt((xs - x1)**2 + (ys - y1)**2)
    else:
        num = np.abs((y2 - y1)*xs - (x2 - x1)*ys + x2*y1 - y2*x1)
        with np.errstate(divide='ignore', invalid='ignore'):  # address 1/0 or 0/0
            dists = num / base_dist
        dists[np.isnan(dists)] = 0.
    index = np.argmax(dists)
    max_dist = dists[index]
    index += 1
    
    with np.errstate(divide='ignore', invalid='ignore'):  # address 1/0 or 0/0
        max_ratio = max_dist / base_dist
    if np.isnan(max_ratio):
        max_ratio = 0.
    
    if max_dist <= distance or max_ratio <= ratio:
        # Distances or distance ratios are not large enough
        return np.asarray([start, end])  # => linear
    
    # Recursively simplify the curve
    first_half = rdpr(points[:index + 1], distance=distance, ratio=ratio)
    second_half = rdpr(points[index:], distance=distance, ratio=ratio)
    
    return np.concat((first_half[:-1], second_half), axis=0)


def pd(points, distance=0.707):  # method 3 (fastest)
    """
    Simplifies a series of points based on the perpendicular distance ratios.
    
    Ref: https://psimpl.sourceforge.net/perpendicular-distance.html
    """
    if points.shape[0] < 3:
        return points
    
    # Get points
    xy0 = points[:-2]  # front points
    xy1 = points[1:-1]  # middle points
    xy2 = points[2:]  # back points
    x0s, y0s = xy0[:, 0], xy0[:, 1]
    x1s, y1s = xy1[:, 0], xy1[:, 1]
    x2s, y2s = xy2[:, 0], xy2[:, 1]
    
    # Calculate the perpendicular distances from middle points to front-back lines
    numerator = np.abs((y2s - y0s)*x1s - (x2s - x0s)*y1s + x2s*y0s - y2s*x0s)
    denominator = np.sqrt((x2s - x0s)**2 + (y2s - y0s)**2)
    with np.errstate(divide='ignore', invalid='ignore'):  # address 1/0 or 0/0
        dists = numerator / denominator
    
    round_trip = denominator == 0  # x0 == x2 and y0 == y2
    xy1_round, xy2_round = xy1[round_trip], xy2[round_trip]
    x1s_round, y1s_round = xy1_round[:, 0], xy1_round[:, 1]
    x2s_round, y2s_round = xy2_round[:, 0], xy2_round[:, 1]
    dists[round_trip] = np.sqrt(
        (x2s_round - x1s_round)**2 + (y2s_round - y1s_round)**2
    )  # from x1 to x2
    
    retain = dists > distance
    
    return np.concat((points[:1], points[1:-1][retain], points[-1:]), axis=0)


def pdr(points, distance=0.707, ratio=0.01):  # method 4
    """
    Simplifies a series of points based on the perpendicular distance ratios.
    
    Ref: https://psimpl.sourceforge.net/perpendicular-distance.html
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
    with np.errstate(divide='ignore', invalid='ignore'):  # address 1/0 or 0/0
        dists = numerator / denominator
        ratios = dists / base_dists
    is_nan = np.isnan(ratios)
    dists[is_nan] = ratios[is_nan] = 0.
    retain = (dists > distance) & (ratios > ratio)
    
    return np.concat((points[:1], points[1:-1][retain], points[-1:]), axis=0)


def antialias(line_id, weight: float = 0.33):
    assert 0 <= weight <= 1, weight
    
    # Mix the foreground and background colors and RGB (0~65535) => RGB (0~255)
    fg = canvas.winfo_rgb(canvas.itemcget(line_id, 'fill'))
    bg = canvas.winfo_rgb(canvas.cget('bg'))
    
    rgb = [  # RGB (0~255)
        int((f*weight + b*(1. - weight)) / 65535. * 255.) for f, b in zip(fg, bg)
    ]
    fill = '#{:02x}{:02x}{:02x}'.format(*rgb)
    
    coords = canvas.coords(line_id)
    width = float(canvas.itemcget(line_id, 'width')) + 1.
    
    # Draw a lighter line
    aa_line_id = canvas.create_line(*coords, fill=fill, width=width)
    canvas.tag_lower(aa_line_id, line_id)


def draw(points):
    canvas.delete('all')
    n0 = points.shape
    t0 = time.monotonic()
    
    points = points.round()
    points = drop_consecutive_duplicates(points)
    n1 = points.shape
    t1 = time.monotonic()
    
    points = pd(points, 0)
    print(points[:10].T)
    points = cutoff(points)
    print(points[:10].T)
    n2 = points.shape
    t2 = time.monotonic()
    
    line_id = canvas.create_line(*points.ravel(), fill='black', width=1)
    antialias(line_id)
    
    t3 = time.monotonic()
    print(f'Drop consecutive duplicates: {n0} points, {t1 - t0: .6f} s')
    print(f'Simplify line: {n1} points, {t2 - t1: .6f} s')
    print(f'Create line: {n2} points, {t3-t2: .6f} s')
    print(f'Total time: {t3 - t0: .6f} s')


root = tk.Tk()
root.wm_geometry('800x600')

canvas = tk.Canvas(root, bg='white')
canvas.pack(fill='both', expand=True)
canvas.bind('<Configure>', lambda e: draw(dense_points))

x = 100 * np.arange(0, 10, 1/10000, dtype=float)
y = 100 * (2 - np.sin(2*np.pi*1/150*x))
#x = np.linspace(50, 80, 10000, endpoint=False)
#y = np.linspace(50, 300, 10000, endpoint=False)
dense_points = np.concat((x[:, None], y[:, None]), axis=1)

root.mainloop()

