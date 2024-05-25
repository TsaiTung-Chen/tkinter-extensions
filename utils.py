#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 19:18:31 2022

@author: tungchentsai
@source: https://github.com/TsaiTung-Chen/tk-utils
"""

from functools import wraps
from typing import Tuple, Union
import tkinter as tk
from tkinter.font import Font
from tkinter import Pack, Grid, Place
from PIL import Image, ImageColor
from PIL.ImageTk import PhotoImage

import numpy as np
import ttkbootstrap as ttk
from ttkbootstrap.utility import scale_size
from ttkbootstrap.colorutils import contrast_color as ttk_contrast_color

from .constants import BUILTIN_WIDGETS, MODIFIERS, MODIFIER_MASKS
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


def get_center_position(widget:tk.BaseWidget) -> Tuple[int, int]:
    widget.update_idletasks()
    if (width := widget.winfo_width()) == 1:
        width = widget.winfo_reqwidth()
    if (height := widget.winfo_height()) == 1:
        height = widget.winfo_reqheight()
    x_root, y_root = widget.winfo_rootx(), widget.winfo_rooty()
    
    return (x_root + width//2, y_root + height//2)


def center_window(to_center:tk.BaseWidget, center_of:tk.BaseWidget):
    x_center, y_center = get_center_position(center_of)
    if (width := to_center.winfo_width()) == 1:
        width = to_center.winfo_reqwidth()
    if (height := to_center.winfo_height()) == 1:
        height = to_center.winfo_reqheight()
    x, y = (x_center - width//2, y_center - height//2)
    tk.Wm.wm_geometry(to_center, f'+{x}+{y}')


def defer(ms:int=1000):
    """Call the decorated function `ms` after the last call on the wrapper.
    Only the last call on the wrapper will exactly execute the decorated 
    function. This means that all the others will be cancelled
    """
    def _wrapper(func):  # decorator
        @wraps(func)
        def _wrapped(*args, **kwargs):
            nonlocal last_id
            root = tk._get_default_root('postpone')
            root.after_cancel(last_id)
            last_id = root.after(ms, lambda: func(*args, **kwargs))
        #
        last_id = -1
        return _wrapped
    #
    return _wrapper


def unbind(widget, sequence, funcid=None):
    """Built-in function don't work as expected that can't unbind the specified 
    function even if `funcid` is provided. So we make this workaround function 
    to replace the use of the original `widget.unbind` method
    """
    if not funcid:
        return widget.unbind(sequence, funcid)
    
    cmds = widget.bind(sequence).split('\n')  # [cmd, '', cmd, '', ...]
    widget.unbind(sequence)  # remove bindings but don't `widget.deletecommand`
    new_cmds = [ c for c in cmds[::2] if funcid not in c ]  # [cmd, cmd, ...]
    if new_cmds:
        widget.bind(sequence, '\n'.join([ c + '\n' for c in new_cmds ]))


def bind_recursively(widget, seqs, funcs, add='', *, key, skip_top_children=True):
    add = '+' if add else ''
    if isinstance(seqs, str):
        seqs = [seqs]
    if callable(funcs):
        funcs = [funcs]
    
    widget._recursively_bound = getattr(widget, "_recursively_bound", dict())
    for seq, func in zip(seqs, funcs):
        assert seq.startswith('<') and seq.endswith('>'), seq
        subbound = widget._recursively_bound.setdefault(seq, dict())
        if key not in subbound:
            subbound[key] = widget.bind(seq, func, add)  # func id
    
    # Propagate
    for child in widget.winfo_children():
        if skip_top_children and (child == child.winfo_toplevel()):
            continue
        bind_recursively(
            child, seqs, funcs, add,
            key=key,
            skip_top_children=skip_top_children
        )


def unbind_recursively(widget, seqs=None, *, key, skip_top_children=True):
    if isinstance(seqs, str):
        seqs = [seqs]
    
    if hasattr(widget, "_recursively_bound"):
        for _seq, subbound in list(widget._recursively_bound.items()):
            for _key, func_id in list(subbound.items()):
                if _key != key:
                    continue
                if seqs is None:
                    unbind(widget, _seq, func_id)
                elif _seq in seqs:
                    unbind(widget, _seq, func_id)
                del subbound[_key]
            if not subbound:
                del widget._recursively_bound[_seq]
        if not widget._recursively_bound:
            del widget._recursively_bound
    
    # Propagate
    for child in widget.winfo_children():
        if skip_top_children and (child == child.winfo_toplevel()):
            continue
        unbind_recursively(
            child, seqs, key=key, skip_top_children=skip_top_children)


def redirect_layout_managers(redirected:tk.BaseWidget,
                             source:tk.BaseWidget,
                             orig_prefix:str='content_'):
    """Redirect layout manager to the outer frame's layout manager
    """
    layout_methods = vars(Pack).keys() | vars(Grid).keys() | vars(Place).keys()
    is_layout = lambda name: (
        (not name.startswith('_')) and
        (name not in ['configure', 'config']) and
        ('rowconfigure' not in name) and ('columnconfigure' not in name) and
        (getattr(type(source), name) is getattr(type(redirected), name))
    )
    
    for name in filter(is_layout, layout_methods):
        setattr(redirected, orig_prefix+name, getattr(redirected, name))
        setattr(redirected, name, getattr(source, name))


def get_modifiers(state:int, platform_specific:bool=True) -> set:
    modifiers = set()
    _modifiers = MODIFIERS if platform_specific else MODIFIER_MASKS
    for mod in _modifiers:
        if state & MODIFIER_MASKS[mod]:
            modifiers.add(mod)
    
    return modifiers


def create_font(new_name:str=None,
                base_font:Union[str, Font]='TkDefaultFont',
                family=None,
                size=None,
                slant=None,
                weight=None) -> Font:
    assert isinstance(base_font, (str, Font)), base_font
    
    if isinstance(base_font, str):
        base_font = tk.font.nametofont(base_font)
    new_font = base_font.copy().actual()
    
    if family is not None:
        new_font["family"] = family
    if size is not None:
        _size, size = size, int(size)
        if isinstance(_size, str) and (_size.startswith('+') or 
                                       _size.startswith('-')):
            size += new_font["size"]
        new_font["size"] = size
    if slant is not None:
        new_font["slant"] = slant
    if weight is not None:
        new_font["weight"] = weight
    
    return Font(name=new_name, **new_font)


def get_font(class_name, default='TkDefaultFont'):
    assert isinstance(class_name, str), class_name
    style = ttk.Style.get_instance()
    font_name = style.lookup(class_name, 'font') or default
    return tk.font.nametofont(font_name)


def create_font_style(class_name:str,
                      prefix:str,
                      bootstyle='',
                      orient=None,
                      apply=True,
                      **kwargs) -> Tuple[str, Font]:
    assert isinstance(class_name, str), class_name
    subs = [ '', *class_name.split('.') ]
    for i, widget_name in enumerate(subs[::-1]):
        if widget_name in BUILTIN_WIDGETS:
            break
    if widget_name != 'Treeview':
        assert widget_name.startswith('T')
        widget_name = widget_name[1:]
    minor = False if i == 0 else True
    
    dummy_widget = getattr(ttk, widget_name)(bootstyle=bootstyle)
    if orient:
        dummy_widget.configure(orient=orient)
    style_name = f'{prefix}.{dummy_widget["style"]}'
    dummy_widget.destroy()
    
    style = ttk.Style.get_instance()
    base_font = get_font(class_name)
    
    if minor:
        minor_style_name = style_name + f'.{subs[-1]}'
        new_font = create_font(minor_style_name, base_font, **kwargs)
        if apply:
            style.configure(style_name)
            style.configure(minor_style_name, font=new_font)
    else:
        new_font = create_font(style_name, base_font, **kwargs)
        if apply:
            style.configure(style_name, font=new_font)
    
    return style_name, new_font


def is_color(color:str):
    try:
        ImageColor.getrgb(color)
    except ValueError:
        return False
    return True


def contrast_color(color:str):
    return ttk_contrast_color(ImageColor.getrgb(color), model='rgb')


def recolor_black(image:Image.Image,
                  new_color:tuple,   # RGB
                  photoimage=False,
                  master:Union[tk.Tk, tk.BaseWidget, None]=None):
    assert isinstance(image, Image.Image), type(image)
    assert isinstance(new_color, tuple), type(new_color)
    
    data = np.array(image.convert('RGBA'), copy=True)
    is_rgb = slice(3)  # channel 0, 1, 2
    is_black = (data[..., is_rgb] == 0).all(axis=-1)  # get 2D pixels (rgb all 0)
    data[is_black, is_rgb] = new_color  # replace black with `new_color`
    img_converted = Image.fromarray(data, mode='RGBA')
    
    if photoimage:
        return PhotoImage(img_converted, master=master)
    return img_converted


def create_image_pair(image:Image.Image,
                      widget:tk.BaseWidget,
                      photoimage=False,
                      master:Union[tk.Tk, tk.BaseWidget, None]=None):
    _convert_bitdepth = lambda x: round((255./65535.) * x)
    
    style = ttk.Style()
    fg = style.lookup(widget["style"], 'foreground')
    fg = tuple(map(_convert_bitdepth, widget.winfo_rgb(fg)))
    fg_pressed = style.lookup(widget["style"], 'foreground', ['pressed'])
    fg_pressed = tuple(map(_convert_bitdepth, widget.winfo_rgb(fg_pressed)))
    img_normal = recolor_black(image, fg, photoimage=photoimage, master=master)
    img_pressed = recolor_black(
        image, fg_pressed, photoimage=photoimage, master=master)
    
    return img_normal, img_pressed


def create_color_image(color,
                       size:Union[int, tuple, list],
                       padding:Union[int, tuple, list]=(0, 0),
                       autoscale=True,
                       photoimage=False,
                       master:Union[tk.Tk, tk.BaseWidget, None]=None):
    """Create an image of size `size` and color `color` with external padding 
    `padding`. The padded margin has the same rgb color but with zero alpha 
    value
    """
    color = ImageColor.getrgb(color)
    assert len(color) in (3, 4), color
    assert isinstance(size, (int, tuple, list)), size
    assert isinstance(padding, (int, tuple, list)), padding
    
    if isinstance(size, int):
        size = [size, size]
    if isinstance(padding, int):
        padding = [padding, padding]
    if len(color) == 3:
        color += (255,)  # RGB => RGBA
    if autoscale:
        master = master or tk._get_default_root('create_color_image')
        size = scale_size(master, size)
    
    sizex, sizey = size
    padx, pady = padding
    padded_size = tuple( s + 2*p for s, p in zip(size, padding) )
    
    # Alpha data
    alpha = np.zeros(padded_size, dtype='uint8')
    alpha[padx:padx + sizex, pady:pady + sizey] = color[-1]
    
    # RGB data
    data = [ np.full(padded_size, c, dtype='uint8') for c in color[:3] ]
    
    # RGBA data
    data.append(alpha)
    data = np.stack(data, axis=-1)
    
    # Convert to RGBA image (row => y, col => x)
    image = Image.fromarray(data.swapaxes(0, 1), mode='RGBA')
    
    if photoimage:
        return PhotoImage(image, master=master)
    return image

