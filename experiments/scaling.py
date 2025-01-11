#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 16:11:40 2024

@author: tungchentsai
"""

import sys
from pathlib import Path
import tkinter as tk
from PIL.ImageTk import PhotoImage

import ttkbootstrap as ttk
# =============================================================================
# ---- Show info for each screen
# =============================================================================
def _get_screen_info_mac() -> float:
    assert platform == 'darwin', platform

    from AppKit import NSScreen  # pyobjc-framework-cocoa
    from Quartz import CGDisplayScreenSize  # pyobjc-framework-quartz
    
    for i, screen in enumerate(NSScreen.screens()):
        description = screen.deviceDescription()
        
        # Physical screen dimensions in millimeters and inches
        screen_no = description["NSScreenNumber"]
        mm_w, mm_h = CGDisplayScreenSize(screen_no)
        in_w, in_h = round(mm_w/25.4, 1), round(mm_h/25.4, 1)
        in_dia = round((in_w**2 + in_h**2)**0.5, 1)  # diagonal
        
        # Logical resolution in dots
        logical_w, logical_h = description["NSDeviceSize"].sizeValue()
        logical_w, logical_h = round(logical_w), round(logical_h)
         # You can find this resolution value in "System Settings > Displays": 
         # "Looks like <horizontal-dots> x <vertical-dots>"
        
        # Raster resolution in PPI (pixels per inch)
        # Note that Apple uses the term DPI for pixels per inch instead
        ppi_x, ppi_y = description["NSDeviceResolution"].sizeValue()
        ppi_x, ppi_y = round(ppi_x), round(ppi_y)
        
        # The backing scale factor, which is used to render the content, in
        # PPD (pixels per dot)
        backing_scale = screen.backingScaleFactor()
         # This value is 2.0 for high-resolution scaled display modes (HiDPI),
         # and 1.0 for all other cases (LoDPI).
         # The backing scaled content stored in frame buffer will then be
         # downscaled to the physical screen resolution with another scaling
         # factor of <physical-pixels> / <rendering-pixels>.
         # So, the final content shown on the screen was first rendered with the
         # backing scale factor and then was downscaled, with another scaling
         # factor in pixels per dot, to the physical screen.
        
        # The backing scale resolution in pixels
        backing_w = round(logical_w * backing_scale)
        backing_h = round(logical_h * backing_scale)
        
        print(f'Display #{i}:')
        print(f'    Physical dimensions = {in_w} × {in_h} ({in_dia}) inches')
        print(f'    Logical resolution = {logical_w} × {logical_h} dots')
        print(f'    Raster resolution = {ppi_x} × {ppi_y} ppi')
        print(f'    Backing scale factor = {backing_scale} pixels per dot')
        print(f'    Backing scale resolution = {backing_w} x {backing_h} pixels')
        
        return backing_scale


def _get_screen_info_else() -> float:
    assert platform != 'darwin', platform
    
    if platform == 'win32':
        from ctypes import windll
        windll.user32.SetProcessDPIAware()
    
    root = tk.Tk()
    root.wm_withdraw()
    
    # System claimed screen size in pixels
    p_w, p_h = root.winfo_screenwidth(), root.winfo_screenheight()

    # System claimed PPI (pixels per inch)
    ppi_sys = root.winfo_fpixels('1i')

    # Tk default PPI
    ppi_tk = 72.

    # Scaling factor
    scale = ppi_sys / ppi_tk  # system PPI / Tk PPI
     # or root.tk.call('tk', 'scaling')
    
    print(f'System claimed screen size = {p_w} x {p_h} pixels')
    print(f'System claimed PPI = {ppi_sys} pixels/inch')
    print(f'Scaling factor (system claimed / Tk default) = {scale}')
    
    root.destroy()
    
    return scale


if (platform := sys.platform) == 'darwin':
    scale = _get_screen_info_mac()
else:
    scale = _get_screen_info_else()


# =============================================================================
# ---- Tk app
# =============================================================================
root = ttk.Window(title='Tk Scaling Test')

text = tk.StringVar()
ttk.Label(root, textvariable=text).pack()

ttk.Button(
    root,
    text='Show scaling factor (~1.33 as Windows default, ~1 in macOS LoDPI mode, ~2 in macOS HiDPI mode)',
    command=lambda: text.set(root.tk.call('tk', 'scaling'))
).pack()

ttk.Button(
    root,
    text='Show system PPI (pixels per inch) (~96 as Windows default, ~72 in macOS LoDPI mode, ~144 in macOS HiDPI mode)',
    command=lambda: text.set(root.winfo_fpixels('1i'))
).pack()

ttk.Button(
    root,
    text='Show screen width in pixels',
    command=lambda: text.set(root.winfo_screenwidth())
).pack()

ttk.Button(
    root,
    text='Set the scaling factor to 0.5',
    command=lambda: text.set(root.tk.call('tk', 'scaling', 0.5))
).pack()

ttk.Button(
    root,
    text='Set the scaling factor to 1.0',
    command=lambda: text.set(root.tk.call('tk', 'scaling', 1.0))
).pack()

ttk.Button(
    root,
    text='Set the scaling factor to 2.0',
    command=lambda: text.set(root.tk.call('tk', 'scaling', 2.0))
).pack()

ttk.Button(
    root,
    text='root.winfo_fpixel(1)',
    command=lambda: text.set(root.winfo_fpixels(1))
).pack()

ttk.Button(
    root,
    text="root.winfo_fpixel('1p')",
    command=lambda: text.set(root.winfo_fpixels('1p'))
).pack()

ttk.Button(
    root,
    text='Create button',
    command=lambda: ttk.Button(root, text='new button').pack()
).pack()

ttk.Button(
    root,
    text='Redraw',
    command=lambda: canvas.destroy() or draw()
).pack()

def draw():
    global canvas, img
    img = PhotoImage(file=Path(__file__).parent / 'data/circle.png')
    canvas = tk.Canvas(root)
    canvas.pack()
    canvas.create_image(0, 0, anchor='nw', image=img)
    canvas.configure(**dict(zip(['width', 'height'], canvas.bbox('all')[2:])))
    return canvas, img

canvas, img = draw()

root.mainloop()

