#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 23:52:18 2024

@author: tungchentsai
"""

import tkinter as tk
from typing import Optional

import ttkbootstrap as ttk
from ttkbootstrap import Colors
# =============================================================================
# ---- Classes
# =============================================================================
class ErrorCatchingWindow(ttk.Window):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._error_message = tk.StringVar(self, name='err_msg')
    
    def report_callback_exception(self, exc, val, tb):
        from traceback import format_exc
        
        super().report_callback_exception(exc, val, tb)
        self._error_message.set(format_exc())


class OptionMenu(ttk.OptionMenu):
    def __init__(self,
                 master,
                 variable=None,
                 values=(),
                 default=None,
                 command=None,
                 direction='below',
                 takefocus=0,
                 style=None,
                 **kwargs):
        super().__init__(master,
                         variable,
                         default,
                         *values,
                         style=style,
                         direction=direction,
                         command=command)
        self.configure(takefocus=takefocus, **kwargs)
    
    def set_command(self, command=None):
        assert command is None or callable(command), command
        
        self._callback = command
        menu = self["menu"]
        max_idx = menu.index('end')
        if max_idx is not None:
            for i in range(max_idx + 1):
                menu.entryconfigure(i, command=command)


class Combobox(ttk.Combobox):
    _scrollbar_fixed: bool = False
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._fix_combobox_scrollbar_bug()
    
    def _fix_combobox_scrollbar_bug(self):
        if Combobox._scrollbar_fixed:
            return
        
        builder = self._root().style._get_builder()
        builder.create_scrollbar_style()
        Combobox._scrollbar_fixed = True
    
    def configure_listbox(self, **kw):
        popdown = self.tk.eval(f'ttk::combobox::PopdownWindow {self}')
        listbox = f'{popdown}.f.l'
        return self.tk.call(listbox, 'configure', *self._options(kw))
    
    def itemconfigure(self, index, **kw):
        popdown = self.tk.eval(f'ttk::combobox::PopdownWindow {self}')
        listbox = f'{popdown}.f.l'
        values = self["values"]
        try:
            self.tk.call(listbox, 'itemconfigure', len(values) - 1)
        except tk.TclError:
            for i, value in enumerate(values):
                self.tk.call(listbox, 'insert', i, value)
        
        return self.tk.call(listbox, 'itemconfigure', index, *self._options(kw))


class ColorButton(ttk.Button):
    @property
    def background(self) -> str:
        return self._background
    
    def __init__(self, master, *args, background: Optional[str] = None, **kw):
        super().__init__(master, *args, **kw)
        if background is None:
            style = ttk.Style.get_instance()
            background = style.configure(self["style"], 'background')
        self._background = background
        self.set_color(background)
        self.bind('<<ThemeChanged>>', lambda e: self.set_color())
    
    def set_color(self, background: Optional[str] = None):
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
    def _keep_changing_color(button: ColorButton, colors: list, ms: int):
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

