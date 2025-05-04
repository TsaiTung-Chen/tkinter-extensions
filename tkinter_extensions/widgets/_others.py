#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 23:52:18 2024

@author: tungchentsai
"""

import tkinter as tk
from typing import Callable
from traceback import format_exc

import ttkbootstrap as ttk
from ttkbootstrap import Colors

from tkinter_extensions import utils
from tkinter_extensions import variables as vrb
# =============================================================================
# ---- Helpers
# =============================================================================
class _StyleBuilderTTK(ttk.style.StyleBuilderTTK):
    def scale_size(self, size):
        return utils.scale_size(self.style.master, size)
    
    def update_combobox_popdown_style(self, *args, **kwargs):
        super().update_combobox_popdown_style(*args, **kwargs)
        
        # Fix combobox' scrollbar
        self.create_scrollbar_style()


# =============================================================================
# ---- Widgets
# =============================================================================
class Window(ttk.Window):
    def __init__(self, *args, **kwargs):
        self._patch_ttkbootstrap()
        super().__init__(*args, **kwargs)
        self._error_message = vrb.StringVar(self, name='err_msg')
    
    def _patch_ttkbootstrap(self):
        # Fix platform-dependent scaling factor
        ttk.utility.scale_size = utils.scale_size
        
        # Patch StyleBuilderTTK
        ttk.style.StyleBuilderTTK = _StyleBuilderTTK
    
    def report_callback_exception(self, exc, val, tb):
        """
        Catch error message.
        """
        super().report_callback_exception(exc, val, tb)
        self._error_message.set(format_exc())


class UndockedFrame(tk.Frame):  # ttk can't be undocked so use tk instead
    def __init__(
            self,
            master,
            *args,
            window_title: str = '',
            dock_callbacks: tuple[Callable | None,
                                  Callable | None] = (None, None),
            undock_callbacks: tuple[Callable | None,
                                    Callable | None] = (None, None),
            undock_button: bool | Callable[[], ttk.Button] = True,
            place_button: bool = True,
            **kwargs
    ):
        assert isinstance(undock_button, (bool, Callable)), undock_button
        
        super().__init__(master, *args, **kwargs)
        self._window_title = window_title
        self._layout_manager: Callable | None = None
        self._layout_info: dict | None = None
        self.set_dock_callbacks(dock_callbacks)
        self.set_undock_callbacks(undock_callbacks)
        
        if undock_button:
            if callable(undock_button):
                assert isinstance(bt:= undock_button(), ttk.Button), bt
                self._undock_button = bt
                self._undock_button.configure(command=self.undock)
            else:
                self._undock_button = bt = ttk.Button(
                    self,
                    text='Undock',
                    takefocus=False,
                    bootstyle='link-primary',
                    command=self.undock
                )
            if place_button:
                self.place_undock_button()
            else:
                bt._place_info = None
            self.bind('<<MapChild>>', lambda e: bt.lift(), add=True)
        else:
            self._undock_button = None
    
    @property
    def undock_button(self):
        return self._undock_button
    
    def place_undock_button(
            self, *, anchor='se', relx=1., rely=1., x=-2, y=-2, **kw):
        assert self._undock_button is not None
        
        self._undock_button.place(
            anchor=anchor, relx=relx, rely=rely, x=x, y=y, **kw)
        self._undock_button._place_info = self._undock_button.place_info()
    
    def set_dock_callbacks(self, callbacks=(None, None)):
        if callbacks is None:
            callbacks = (None, None)
        assert len(callbacks) == 2, callbacks
        assert all( c is None or callable(c) for c in callbacks ), callbacks
        self._dock_callbacks = callbacks
    
    def set_undock_callbacks(self, callbacks=(None, None)):
        if callbacks is None:
            callbacks = (None, None)
        assert len(callbacks) == 2, callbacks
        assert all( c is None or callable(c) for c in callbacks ), callbacks
        self._undock_callbacks = callbacks
    
    def undock(self):
        if (manager := self.winfo_manager()) == 'pack':
            self._layout_manager = self.pack
            self._layout_info = self.pack_info()
        elif manager == 'grid':
            self._layout_manager = self.grid
            self._layout_info = self.grid_info()
        elif manager == 'place':
            self._layout_manager = self.place
            self._layout_info = self.place_info()
        else:
            raise RuntimeError(
                f"Unknown layout manager: {repr(manager)}. Should be any of "
                "'pack', 'grid', or 'place'."
            )
        
        callback_begin, callback_final = self._undock_callbacks
        
        if callback_begin:
            callback_begin()
        
        tk.Wm.wm_manage(self, self)  # make self frame become a toplevel
        tk.Wm.wm_withdraw(self)
        tk.Wm.wm_title(self, self._window_title)
        tk.Wm.wm_protocol(self, 'WM_DELETE_WINDOW', self.dock)
        
        if self._undock_button and self._undock_button._place_info:
            self._undock_button.place_forget()
        
        self._root().focus_set()
        self.focus_set()
        
        if callback_final:
            callback_final()
        
        tk.Wm.wm_deiconify(self)
        self.lift()
    
    def dock(self):
        callback_begin, callback_final = self._dock_callbacks
        
        if callback_begin:
            callback_begin()
        
        tk.Wm.wm_forget(self, self)
        self._layout_manager(**self._layout_info)
        
        if self._undock_button and self._undock_button._place_info:
            self._undock_button.place(**self._undock_button._place_info)
        
        self._root().focus_set()
        self.focus_set()
        
        if callback_final:
            callback_final()


class OptionMenu(ttk.OptionMenu):
    def __init__(
            self,
            master,
            variable=None,
            values=(),
            default=None,
            command=None,
            direction='below',
            takefocus=False,
            style=None,
            **kwargs
    ):
        super().__init__(
            master,
            variable,
            default,
            *values,
            style=style,
            direction=direction,
            command=command
        )
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
    
    def __init__(self, master, *args, background: str | None = None, **kw):
        super().__init__(master, *args, **kw)
        if background is None:
            style = self._root().style
            background = style.configure(self["style"], 'background')
        self._background = background
        self.set_color(background)
        self.bind('<<ThemeChanged>>', lambda e: self.set_color(), add=True)
    
    def set_color(self, background: str | None = None):
        """Ref: `ttkbootstrap.style.StyleBuilderTTK.create_button_style`
        """
        style_name = f'{id(self)}.TButton'
        style = self._root().style
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


class WrapLabel(ttk.Label):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bind(
            '<Configure>', lambda e: self.configure(wraplength=e.width), add=True
        )


class ToolTip(ttk.tooltip.ToolTip):
    def __init__(
        self,
        widget,
        text='widget info',
        bootstyle=None,
        wraplength=None,
        delay=250,   # milliseconds
        padding=(4, 1),
        **kwargs,
    ):
        super().__init__(
            widget=widget,
            text=text,
            bootstyle=bootstyle,
            wraplength=wraplength,
            delay=delay,
            **kwargs
        )
        
        if wraplength is None:
            self.wraplength = wraplength  # override the value
        self.padding = padding  # save for later use
    
    def show_tip(self, *_):
        if self.toplevel:
            return
        
        super().show_tip(*_)
        
        lb, = self.toplevel.winfo_children()
        assert isinstance(lb, ttk.Label), (type(lb), lb)
        
        lb.configure(padding=self.padding)  # override the value set in super func


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
    
    def _new_error():
        def _raise_error():
            raise ValueError("Error Catched")
        
        error_lb.configure(text='No error')
        root.after(1000, _raise_error)
    #
    
    root = Window(size=(800, 600))
    
    ttk.Button(root, text='Raise error', command=_new_error).pack(pady=[3, 0])
    error_lb = ttk.Label(root, text='No error', bootstyle='inverse')
    error_lb.pack(pady=[1, 3])
    root._error_message.trace_add(
        'write',
        lambda name, *_: error_lb.configure(text=root.getvar(name))
    )
    
    ttk.Button(root, text='Normal Button', bootstyle='danger').pack(pady=[3, 0])
    
    red_bt = ColorButton(root, text='Red Button')
    red_bt.pack(pady=[1, 0])
    red_bt.set_color('red')
    
    varying_bt = ColorButton(root, text='Varying Button')
    varying_bt.pack(pady=[1, 3])
    varying_bt.set_color('red')
    
    colors = ['red', 'blue', 'green']
    root.after(1500, _keep_changing_color, varying_bt, colors, 1500)
    root.after(10000, lambda: root.style.theme_use('cyborg'))
    
    root.mainloop()

