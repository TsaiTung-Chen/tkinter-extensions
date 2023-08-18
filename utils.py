#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 21:25:08 2023

@author: tungchentsai
"""

import tkinter as tk
from tkinter import Pack, Grid, Place
from typing import Tuple

from .constants import MODIFIERS, MODIFIER_MASKS
# =============================================================================
# ---- Functions
# =============================================================================
def quit_if_all_closed(window):
    def _wrapped(event=None):
        root = window._root()
        if len(root.children) > 1:
            window.destroy()
        else:
            root.quit()
            root.destroy()
    return _wrapped


def redirect_layout_managers(redirected, target, orig_prefix='content_'):
    # Redirect layout manager to the outer frame's layout manager
    layout_methods = vars(Pack).keys() | vars(Grid).keys() | vars(Place).keys()
    is_layout = lambda name: (
        (not name.startswith('_')) and
        (name not in ['configure', 'config']) and
        ('rowconfigure' not in name) and ('columnconfigure' not in name) and
        (getattr(type(redirected), name) is getattr(type(target), name))
    )
    
    for name in filter(is_layout, layout_methods):
        setattr(redirected, orig_prefix+name, getattr(redirected, name))
        setattr(redirected, name, getattr(target, name))


def bind_recursively(widget, seqs, funcs, add=''):
    add = '+' if add else ''
    if isinstance(seqs, str):
        seqs = [seqs]
    if not isinstance(funcs, (list, tuple)):
        funcs = [funcs]
    
    widget._bound = getattr(widget, "_bound", dict())
    for seq, func in zip(seqs, funcs):
        assert seq.startswith('<') and seq.endswith('>'), seq
        if seq not in widget._bound:
            widget._bound[seq] = widget.bind(seq, func, add)  # func id
    
    # Propagate
    for child in widget.winfo_children():
        bind_recursively(child, seqs, funcs)


def unbind_recursively(widget, seqs=None):
    if isinstance(seqs, str):
        seqs = [seqs]
    
    if getattr(widget, "_bound", None):
        for seq, func_id in list(widget._bound.items()):
            if (seqs is None) or (seq in seqs):
                widget.unbind(seq, func_id)
                del widget._bound[seq]
    
    # Propagate
    for child in widget.winfo_children():
        unbind_recursively(child)


def get_modifiers(state:int, platform_specific:bool=True) -> set:
    modifiers = set()
    _modifiers = MODIFIERS if platform_specific else MODIFIER_MASKS
    for mod in _modifiers:
        if state & MODIFIER_MASKS[mod]:
            modifiers.add(mod)
    
    return modifiers


def get_center_position(widget:tk.BaseWidget) -> Tuple[int, int]:
    widget.update_idletasks()
    width, height = widget.winfo_width(), widget.winfo_height()
    x_root, y_root = widget.winfo_rootx(), widget.winfo_rooty()
    
    return (x_root + width//2, y_root + height//2)


def center_window(to_center:tk.BaseWidget, center_of:tk.BaseWidget):
    x_center, y_center = get_center_position(center_of)
    width, height = to_center.winfo_width(), to_center.winfo_height()
    x, y = (x_center - width//2, y_center - height//2)
    tk.Wm.wm_geometry(to_center, f'+{x}+{y}')

