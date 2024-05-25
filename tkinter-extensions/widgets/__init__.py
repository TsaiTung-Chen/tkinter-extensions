#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 22:35:24 2023

@author: tungchentsai
"""
from typing import Optional

import ttkbootstrap as ttk
from ttkbootstrap import Colors

from .dnd import OrderlyContainer, TriggerOrderlyContainer
from .collapsed import CollapsedFrame
from .scrolled import ScrolledWidget
from .scrolled import ScrolledFrame, ScrolledLabelframe
from .scrolled import ScrolledText, ScrolledTreeview
from .undocked import UndockedFrame
from .plotters import BasePlotter
from .spreadsheets import Sheet, Book
# =============================================================================
# ---- Classes
# =============================================================================
class ColorButton(ttk.Button):
    @property
    def background(self) -> str:
        return self._background
    
    def __init__(self, master, *args, background:Optional[str]=None, **kw):
        super().__init__(master, *args, **kw)
        if background is None:
            style = ttk.Style.get_instance()
            background = style.configure(self["style"], 'background')
        self._background = background
        self.set_color(background)
        self.bind('<<ThemeChanged>>', lambda e: self.set_color())
    
    def set_color(self, background:Optional[str]=None):
        """Ref: `ttkbootstrap.style.StyleBuilderTTK.create_button_style`
        """
        style_name = f'{id(self)}.TButton'
        style = ttk.Style.get_instance()
        colors = style.colors

        bordercolor = background = background or self._background
        disabled_bg = Colors.make_transparent(0.1, colors.fg, colors.bg)
        disabled_fg = Colors.make_transparent(0.3, colors.fg, colors.bg)
        pressed = Colors.make_transparent(0.6, background, colors.bg)
        hover = Colors.make_transparent(0.7, background, colors.bg)        
        
        style._build_configure(
            style_name,
            background=background,
            bordercolor=bordercolor,
            darkcolor=background,
            lightcolor=background
        )
        style.map(
            style_name,
            foreground=[("disabled", disabled_fg)],
            background=[
                ("disabled", disabled_bg),
                ("pressed !disabled", pressed),
                ("hover !disabled", hover),
            ],
            bordercolor=[("disabled", disabled_bg)],
            darkcolor=[
                ("disabled", disabled_bg),
                ("pressed !disabled", pressed),
                ("hover !disabled", hover),
            ],
            lightcolor=[
                ("disabled", disabled_bg),
                ("pressed !disabled", pressed),
                ("hover !disabled", hover),
            ],
        )
        style._register_ttkstyle(style_name)
        
        self.configure(style=style_name)
        self._background = background
        
        return self._background


# =============================================================================
# ---- Main
# =============================================================================
if __name__ == '__main__':
    def _keep_changing_color(button:ColorButton, colors:list, ms:int):
        colors = list(colors)
        old_color = button._background
        old_idx = colors.index(old_color)
        new_idx = old_idx + 1
        if new_idx >= len(colors):
            new_idx = 0
        new_color = colors[new_idx]
        button.set_color(new_color)
        button.after(ms, _keep_changing_color, button, colors, ms)
    #
    root = ttk.Window(size=(1280, 960))
    ttk.Button(root, text='Normal Button', bootstyle='danger').pack()
    red_bt = ColorButton(root, text='Red Button')
    red_bt.pack()
    red_bt.set_color('red')
    varying_bt = ColorButton(root, text='Red Button')
    varying_bt.pack()
    varying_bt.set_color('red')
    colors = ['red', 'blue', 'green']
    root.after(1500, _keep_changing_color, varying_bt, colors, 1500)
    root.after(10000, lambda: root.style.theme_use('cyborg'))
    root.mainloop()

